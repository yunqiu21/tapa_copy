#include "type.h"

#include <regex>
#include <string>

#include "clang/AST/AST.h"

using std::regex;
using std::regex_match;
using std::string;

using clang::RecordDecl;

bool IsTlpType(const RecordDecl* decl, const string& type_name) {
  return decl != nullptr && regex_match(decl->getQualifiedNameAsString(),
                                        regex{"tlp::" + type_name});
}