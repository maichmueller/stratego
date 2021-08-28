
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_logic(py::module_ &);

PYBIND11_MODULE(_core, m)
{
    init_logic(m);
}