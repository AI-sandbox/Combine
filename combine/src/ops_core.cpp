#include "decl.hpp"
#include <combine_core/core.hpp>
#include <combine_core/util/types.hpp>

namespace cb = combine_core;

PYBIND11_MODULE(combine_core, m) {
    auto m_io = m.def_submodule("io", "IO submodule.");
    register_io(m_io);

    m.def("to_sample_major_int8", &cb::to_sample_major<int8_t>); 
    m.def("calldata_sum_int8", &cb::calldata_sum<int8_t>); 
    m.def("column_mean", &cb::column_mean<int8_t, double>);
    m.def("calldata_subset_rows_cols_int8", &cb::calldata_subset_rows_cols<int8_t, int32_t>);
}