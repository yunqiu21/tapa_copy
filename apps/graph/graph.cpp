#include <cassert>
#include <cstring>

#include <tlp.h>

using Vid = uint32_t;
using Eid = uint32_t;
using Pid = uint16_t;

using VertexAttr = Vid;

using std::ostream;

struct Edge {
  Vid src;
  Vid dst;
};

struct TaskReq {
  enum Phase { kScatter = 0, kGather = 1 };
  Phase phase;
  bool IsScatter() const { return phase == kScatter; }
  bool IsGather() const { return phase == kGather; }
  Pid pid;
  Vid base_vid;
  Vid num_vertices;
  Eid num_edges;
  Vid vid_offset;
  Eid eid_offset;
};

ostream& operator<<(ostream& os, const TaskReq::Phase& obj) {
  return os << (obj == TaskReq::kScatter ? "SCATTER" : "GATHER");
}

ostream& operator<<(ostream& os, const TaskReq& obj) {
  return os << "{phase: " << obj.phase << ", pid: " << obj.pid
            << ", base_vid: " << obj.base_vid
            << ", num_vertices: " << obj.num_vertices
            << ", num_edges: " << obj.num_edges
            << ", vid_offset: " << obj.vid_offset
            << ", eid_offset: " << obj.eid_offset << "}";
}

struct TaskResp {
  TaskReq::Phase phase;
  Pid pid;
  bool active;
};

ostream& operator<<(ostream& os, const TaskResp& obj) {
  return os << "{phase: " << obj.phase << ", pid: " << obj.pid
            << ", active: " << obj.active << "}";
}

struct Update {
  Vid dst;
  Vid value;
};

ostream& operator<<(ostream& os, const Update& obj) {
  return os << "{dst: " << obj.dst << ", value: " << obj.value << "}";
}

struct UpdateConfig {
  enum Item { kBaseVid = 0, kPartitionSize = 1, kUpdateOffset = 2 };
  Item item;
  Vid vid;
  Eid eid;
};

ostream& operator<<(ostream& os, const UpdateConfig::Item& obj) {
  switch (obj) {
    case UpdateConfig::kBaseVid:
      os << "BASE_VID";
      break;
    case UpdateConfig::kPartitionSize:
      os << "PARTITION_SIZE";
      break;
    case UpdateConfig::kUpdateOffset:
      os << "UPDATE_OFFSET";
      break;
  }
  return os;
}

ostream& operator<<(ostream& os, const UpdateConfig& obj) {
  os << "{item: " << obj.item;
  switch (obj.item) {
    case UpdateConfig::kBaseVid:
    case UpdateConfig::kPartitionSize:
      os << ", vid: " << obj.vid;
      break;
    case UpdateConfig::kUpdateOffset:
      os << ", eid: " << obj.eid;
      break;
  }
  return os << "}";
}

struct UpdateReq {
  TaskReq::Phase phase;
  Pid pid;
};

ostream& operator<<(ostream& os, const UpdateReq& obj) {
  return os << "{phase: " << obj.phase << ", pid: " << obj.pid << "}";
}

const int kMaxNumPartitions = 1024;
const int kMaxPartitionSize = 1024 * 1024;

void Control(Pid num_partitions, const Vid* num_vertices, const Eid* num_edges,
             tlp::stream<UpdateConfig>& update_config_q,
             tlp::stream<TaskReq>& req_q, tlp::stream<TaskResp>& resp_q) {
  // Keeps track of all partitions.

  // Vid of the 0-th vertex in each partition.
  Vid base_vids[kMaxNumPartitions];
  // Number of vertices in each partition.
  Vid num_vertices_local[kMaxNumPartitions];
  // Number of edges in each partition.
  Eid num_edges_local[kMaxNumPartitions];
  // Memory offset of the 0-th vertex in each partition.
  Vid vid_offsets[kMaxNumPartitions];
  // Memory offset of the 0-th edge in each partition.
  Eid eid_offsets[kMaxNumPartitions];

  Vid base_vid_acc = num_vertices[0];
  Vid vid_offset_acc = 0;
  Eid eid_offset_acc = 0;
  bool done[kMaxNumPartitions] = {};
  for (Pid pid = 0; pid < num_partitions; ++pid) {
    Vid num_vertices_delta = num_vertices[pid + 1];
    Eid num_edges_delta = num_edges[pid];

    base_vids[pid] = base_vid_acc;
    num_vertices_local[pid] = num_vertices_delta;
    num_edges_local[pid] = num_edges_delta;
    vid_offsets[pid] = vid_offset_acc;
    eid_offsets[pid] = eid_offset_acc;

    base_vid_acc += num_vertices_delta;
    vid_offset_acc += num_vertices_delta;
    eid_offset_acc += num_edges_delta;
  }

  // Initialize UpdateHandler, needed only once per execution.
  update_config_q.write({UpdateConfig::kBaseVid, base_vids[0], 0});
  update_config_q.write(
      {UpdateConfig::kPartitionSize, num_vertices_local[0], 0});
  for (Pid pid = 0; pid < num_partitions; ++pid) {
    VLOG(8) << "info@Control: eid offset[" << pid
            << "]: " << eid_offset_acc * pid;
    UpdateConfig info{UpdateConfig::kUpdateOffset, 0, eid_offset_acc * pid};
    update_config_q.write(info);
  }
  update_config_q.close();

  bool all_done = false;
  while (!all_done) {
    all_done = true;

    // Do the scatter phase for each partition, if active.
    for (Pid pid = 0; pid < num_partitions; ++pid) {
      if (!done[pid]) {
        TaskReq req{TaskReq::kScatter,    pid,
                    base_vids[pid],       num_vertices_local[pid],
                    num_edges_local[pid], vid_offsets[pid],
                    eid_offsets[pid]};
        req_q.write(req);
      }
    }

    // Wait until all partitions are done with the scatter phase.
    for (Pid pid = 0; pid < num_partitions; ++pid) {
      if (!done[pid]) {
        TaskResp resp = resp_q.read();
        assert(resp.phase == TaskReq::kScatter);
      }
    }

    // Do the gather phase for each partition.
    for (Pid pid = 0; pid < num_partitions; ++pid) {
      TaskReq req{TaskReq::kGather,     pid,
                  base_vids[pid],       num_vertices_local[pid],
                  num_edges_local[pid], vid_offsets[pid],
                  eid_offsets[pid]};
      req_q.write(req);
    }

    // Wait until all partitions are done with the gather phase.
    for (Pid pid = 0; pid < num_partitions; ++pid) {
      TaskResp resp = resp_q.read();
      assert(resp.phase == TaskReq::kGather);
      VLOG(3) << "recv@Control: " << resp;
      if (resp.active) {
        all_done = false;
      } else {
        done[pid] = true;
      }
    }
    VLOG(3) << "info@Control: " << (all_done ? "" : "not ") << "all done";
  }

  // Terminates the ProcElem.
  req_q.close();
}

void UpdateHandler(Pid num_partitions,
                   tlp::stream<UpdateConfig>& update_config_q,
                   tlp::stream<UpdateReq>& update_req_q,
                   tlp::stream<Update>& update_in_q,
                   tlp::stream<Update>& update_out_q, Update* updates) {
  // Base vid of all vertices; used to determine dst partition id.
  Vid base_vid = 0;
  // Used to determine dst partition id.
  Vid partition_size = 1;
  // Memory offsets of each update partition.
  Eid update_offsets[kMaxNumPartitions] = {};
  // Number of updates of each update partition in memory.
  Eid num_updates[kMaxNumPartitions] = {};

  // Initialization; needed only once per execution.
  int update_offset_idx = 0;
  while (!update_config_q.eos()) {
    auto config = update_config_q.read();
    VLOG(5) << "recv@UpdateHandler: UpdateConfig: " << config;
    switch (config.item) {
      case UpdateConfig::kBaseVid:
        base_vid = config.vid;
        break;
      case UpdateConfig::kPartitionSize:
        partition_size = config.vid;
        break;
      case UpdateConfig::kUpdateOffset:
        update_offsets[update_offset_idx] = config.eid;
        ++update_offset_idx;
        break;
    }
  }

  while (!update_req_q.eos()) {
    // Each UpdateReq either requests forwarding all Updates from ProcElem to
    // the memory (scatter phase), or requests forwarding all Updates from the
    // memory to ProcElem (gather phase).
    const auto update_req = update_req_q.read();
    VLOG(5) << "recv@UpdateHandler: UpdateReq: " << update_req;
    if (update_req.phase == TaskReq::kScatter) {
      while (!update_in_q.eos()) {
        Update update = update_in_q.read();
        VLOG(5) << "recv@UpdateHandler: Update: " << update;
        Pid pid = (update.dst - base_vid) / partition_size;
        VLOG(5) << "info@UpdateHandler: dst partition id: " << pid;
        Eid update_idx = num_updates[pid];
        Eid update_offset = update_offsets[pid] + update_idx;

        updates[update_offset] = update;

        num_updates[pid] = update_idx + 1;
      }
      update_in_q.open();
    } else {
      const auto pid = update_req.pid;
      VLOG(6) << "info@UpdateHandler: num_updates[" << pid
              << "]: " << num_updates[pid];
      for (Eid update_idx = 0; update_idx < num_updates[pid]; ++update_idx) {
        Eid update_offset = update_offsets[pid] + update_idx;
        VLOG(5) << "send@UpdateHandler: update_offset: " << update_offset
                << "Update: " << updates[update_offset];
        update_out_q.write(updates[update_offset]);
      }
      num_updates[pid] = 0;  // Reset for the next scatter phase.
      update_out_q.close();
    }
  }
  VLOG(3) << "info@UpdateHandler: done";
}

void ProcElem(tlp::stream<TaskReq>& req_q, tlp::stream<TaskResp>& resp_q,
              tlp::stream<UpdateReq>& update_req_q,
              tlp::stream<Update>& update_in_q,
              tlp::stream<Update>& update_out_q, VertexAttr* vertices,
              const Edge* edges) {
  VertexAttr vertices_local[kMaxPartitionSize];
  while (!req_q.eos()) {
    const TaskReq req = req_q.read();
    VLOG(5) << "recv@ProcElem: TaskReq: " << req;
    update_req_q.write({req.phase, req.pid});
    memcpy(vertices_local, vertices + req.vid_offset,
           req.num_vertices * sizeof(VertexAttr));
    bool active = false;
    if (req.IsScatter()) {
      for (Eid eid = 0; eid < req.num_edges; ++eid) {
        auto edge = edges[req.eid_offset + eid];
        auto vertex_attr = vertices_local[edge.src - req.base_vid];
        Update update;
        update.dst = edge.dst;
        update.value = vertex_attr;
        VLOG(5) << "send@ProcElem: Update: " << update;
        update_out_q.write(update);
      }
      update_out_q.close();
    } else {
      while (!update_in_q.eos()) {
        auto update = update_in_q.read();
        VLOG(5) << "recv@ProcElem: Update: " << update;
        auto old_vertex_value = vertices_local[update.dst - req.base_vid];
        if (update.value < old_vertex_value) {
          vertices_local[update.dst - req.base_vid] = update.value;
          active = true;
        }
      }
      update_in_q.open();
      memcpy(vertices + req.vid_offset, vertices_local,
             req.num_vertices * sizeof(VertexAttr));
    }
    TaskResp resp{req.phase, req.pid, active};
    resp_q.write(resp);
  }

  // Terminates the UpdateHandler.
  update_req_q.close();
}

void Graph(Pid num_partitions, const Vid* num_vertices, const Eid* num_edges,
           VertexAttr* vertices, const Edge* edges, Update* updates) {
  tlp::stream<TaskReq, kMaxNumPartitions> task_req("task_req");
  tlp::stream<TaskResp, 1> task_resp("task_resp");
  tlp::stream<Update, 1> update_pe2handler("update_pe2handler");
  tlp::stream<Update, 1> update_handler2pe("update_handler2pe");
  tlp::stream<UpdateConfig, 1> update_config("update_config");
  tlp::stream<UpdateReq, 1> update_req("update_req");

  tlp::task()
      .invoke<0>(Control, num_partitions, num_vertices, num_edges,
                 update_config, task_req, task_resp)
      .invoke<0>(UpdateHandler, num_partitions, update_config, update_req,
                 update_pe2handler, update_handler2pe, updates)
      .invoke<0>(ProcElem, task_req, task_resp, update_req, update_handler2pe,
                 update_pe2handler, vertices, edges);
}