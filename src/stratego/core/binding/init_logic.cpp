#include <pybind11/pybind11.h>
#include "stratego/logic.h"

namespace py = pybind11;

void init_logic(py::module_ &m) {
    py::class_< Logic > logic(m, "LogicCPP");
    logic.def(py::init());

}