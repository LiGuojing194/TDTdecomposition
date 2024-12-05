#include <torch/extension.h>

//********************************segment add********************************************************//
void segment_add_cpu(const int *col_indices, const int *col_ptr, int *output, const int numRows);
void segment_add_cuda(const int *col_indices, const int *col_ptr, int *output, const int numRows);


void segment_add(torch::Tensor col_indices, torch::Tensor col_ptr, torch::Tensor output) {
    // 确保输入是整型且在 CUDA 设备上
    if (col_ptr.is_cuda()){
        segment_add_cuda(col_indices.data_ptr<int>(), col_ptr.data_ptr<int>(), output.data_ptr<int>(), output.size(0));
        }
    else{
        segment_add_cpu(col_indices.data_ptr<int>(), col_ptr.data_ptr<int>(), output.data_ptr<int>(), output.size(0));
        }   
}

//********************************two ptr segment isin********************************************************//
// 声明 CUDA 函数
void checkElementsTwoPtr_cuda(const torch::Tensor a_rows, const torch::Tensor b_col_indices, const torch::Tensor b_row_ptr, torch::Tensor output);
void segment_isin2_cpu(const int *a_rows, const int *b_col_indices, const int *b_row_ptr, int *output, const int numRows);

// 包装函数
void segment_isin2(torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    torch::Tensor output
) {
    if (a_rows.is_cuda()){
    // 调用 CUDA 函数
    checkElementsTwoPtr_cuda(a_rows, b_col_indices, b_row_ptr, output);}
    else{
    segment_isin2_cpu(a_rows.data_ptr<int>(), b_col_indices.data_ptr<int>(), b_row_ptr.data_ptr<int>(), output.data_ptr<int>(), output.size(0));}
}
//****************************two segment isin tile*************************************************************//
// 声明 CUDA 函数
void checkElementsTwoPtrTile_cuda(const torch::Tensor a_rows, const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr, const int n_cut, torch::Tensor output);
void segment_isin2tile_cpu(const int *a_rows, const int *b_col_indices, const int *b_row_ptr, const int n_cut, 
    int *output, const int numRows);

// 包装函数
void segment_isin2tile(
    torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    int n_cut,
    torch::Tensor output
) {
    if (a_rows.is_cuda()){
        // 调用 CUDA 函数
        checkElementsTwoPtrTile_cuda(a_rows, b_col_indices, b_row_ptr, n_cut, output);}
    else{
        segment_isin2tile_cpu(a_rows.data_ptr<int>(), b_col_indices.data_ptr<int>(), b_row_ptr.data_ptr<int>(), n_cut, output.data_ptr<int>(), output.size(0)*n_cut);
    }
}
//****************************sub affect_support *************************************************************//
void AllAffectedSupport_cuda(const torch::Tensor e_affect, const torch::Tensor a_rows,
    const torch::Tensor b_col_indices, const torch::Tensor b_row_ptr,
    const torch::Tensor mark, const int l, 
    torch::Tensor n_mark, torch::Tensor output);

void AllAffectedSupport_cpu(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows);
 
void sub_AllAffectedSupport(
    torch::Tensor e_affect,
    torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    torch::Tensor mark,
    int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
){
    if (a_rows.is_cuda()){
        AllAffectedSupport_cuda(e_affect, a_rows, b_col_indices, b_row_ptr, mark, l,  n_mark, output);}
    else{
        AllAffectedSupport_cpu(e_affect.data_ptr<int>(), a_rows.data_ptr<int>(), b_col_indices.data_ptr<int>(), b_row_ptr.data_ptr<int>(), mark.data_ptr<bool>(), l,
          n_mark.data_ptr<bool>(), output.data_ptr<int>(), e_affect.size(0));
    }
}
//////////////////////*******************not**************************/////////////////
void AllAffectedSupport_not_cuda(const torch::Tensor e_affect, const torch::Tensor a_rows,
    const torch::Tensor b_col_indices, const torch::Tensor b_row_ptr,
    const torch::Tensor mark, const int l, 
    torch::Tensor n_mark, torch::Tensor output);
 
void sub_AllAffectedSupport_not(
    torch::Tensor e_affect,
    torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    torch::Tensor mark,
    int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
){
    // if (a_rows.is_cuda()){
        AllAffectedSupport_not_cuda(e_affect, a_rows, b_col_indices, b_row_ptr, mark, l,  n_mark, output);
    // }
}
//////////////////////*******************not win**************************/////////////////
void AllAffectedSupport_notwin_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output,
    const int windows_high, 
    torch::Tensor in_windows_mask, 
    torch::Tensor buff_index, 
    torch::Tensor buf_size
);
 
void sub_AllAffectedSupport_notwin(
    torch::Tensor e_affect,
    torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    torch::Tensor mark,
    int l, 
    torch::Tensor n_mark, 
    torch::Tensor output,
    int windows_high, 
    torch::Tensor in_windows_mask, 
    torch::Tensor buff_index, 
    torch::Tensor buf_size
){
    // if (a_rows.is_cuda()){
        AllAffectedSupport_notwin_cuda(e_affect, a_rows, b_col_indices, b_row_ptr, mark, l,  n_mark, output, windows_high, in_windows_mask, buff_index, buf_size);
    // }
}
// //****************************sub affect_support tile  *************************************************************//
void AllAffectedSupport_tile_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const int n_cut,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
);

void AllAffectedSupport_tile_cpu(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut,  const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows
);
 
void sub_AllAffectedSupport_tile(
    torch::Tensor e_affect,
    torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    int n_cut,
    torch::Tensor mark,
    int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
){
    if (a_rows.is_cuda()){
        AllAffectedSupport_tile_cuda(e_affect, a_rows, b_col_indices, b_row_ptr, n_cut, mark, l,  n_mark, output);}
    else{
        AllAffectedSupport_tile_cpu(e_affect.data_ptr<int>(), a_rows.data_ptr<int>(), b_col_indices.data_ptr<int>(), b_row_ptr.data_ptr<int>(), n_cut, mark.data_ptr<bool>(), l,  n_mark.data_ptr<bool>(), output.data_ptr<int>(), e_affect.size(0)*n_cut);}
}
//////////////////////*******************not**************************/////////////////
void AllAffectedSupport_tilenot_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const int n_cut,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
);
 
void sub_AllAffectedSupport_tilenot(
    torch::Tensor e_affect,
    torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    int n_cut,
    torch::Tensor mark,
    int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
){
    // if (a_rows.is_cuda()){
        AllAffectedSupport_tilenot_cuda(e_affect, a_rows, b_col_indices, b_row_ptr, n_cut, mark, l,  n_mark, output);
    // }
}

//////////////////////*******************not windows**************************/////////////////
void AllAffectedSupport_tilenotwin_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const int n_cut,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output,
    const int windows_high, 
    torch::Tensor in_windows_mask, 
    torch::Tensor buff_index, 
    torch::Tensor buf_size
);
 
void sub_AllAffectedSupport_tilenotwin(
    torch::Tensor e_affect,
    torch::Tensor a_rows,
    torch::Tensor b_col_indices,
    torch::Tensor b_row_ptr,
    int n_cut,
    torch::Tensor mark,
    int l, 
    torch::Tensor n_mark, 
    torch::Tensor output,
    int windows_high, 
    torch::Tensor in_windows_mask, 
    torch::Tensor buff_index, 
    torch::Tensor buf_size
){
    // if (a_rows.is_cuda()){
        AllAffectedSupport_tilenotwin_cuda(e_affect, a_rows, b_col_indices, b_row_ptr, n_cut, mark, l,  n_mark, output, windows_high, in_windows_mask, buff_index, buf_size);
    // }
}
//********************************two ptr segment isinis_m********************************************************//
// 声明 CUDA 函数
void segment_isinis_m_cuda(
    const torch::Tensor a_columns,
    const torch::Tensor a_row_ptr,
    const torch::Tensor b_columns,
    const torch::Tensor b_row_ptr,
    torch::Tensor mask1,
    torch::Tensor mask2);
// void segment_isin2_cpu(const int *a_rows, const int *b_col_indices, const int *b_row_ptr, int *output, const int numRows);

// 包装函数
void segment_isinis_m(torch::Tensor a_columns,
    torch::Tensor a_row_ptr,
    torch::Tensor b_columns,
    torch::Tensor b_row_ptr,
    torch::Tensor mask1,
    torch::Tensor mask2
) {
    // if (a_columns.is_cuda()){
    // 调用 CUDA 函数
    segment_isinis_m_cuda(a_columns, a_row_ptr, b_columns, b_row_ptr, mask1, mask2);
    // }
    // else{
    // segment_isin2_cpu(a_rows.data_ptr<int>(), b_col_indices.data_ptr<int>(), b_row_ptr.data_ptr<int>(), output.data_ptr<int>(), output.size(0));}
}

//****************************sub_suppport_affect(left_nbr, right_nbr, e_curr_rep, support, e_affect_mask)***************//
// 声明 CUDA 函数
void sub_suppport_affect_cuda(
    const torch::Tensor left_nbr,
    const torch::Tensor right_nbr,
    const torch::Tensor e_curr_rep,
    torch::Tensor support,
    const torch::Tensor mask1, 
    const torch::Tensor mask2,
    torch::Tensor e_affect_mask,
    const int l);
// void segment_isin2_cpu(const int *a_rows, const int *b_col_indices, const int *b_row_ptr, int *output, const int numRows);

// 包装函数
void sub_suppport_affect(torch::Tensor left_nbr,
    torch::Tensor right_nbr,
    torch::Tensor e_curr_rep,
    torch::Tensor support,
    torch::Tensor mask1, 
    torch::Tensor mask2,
    torch::Tensor e_affect_mask,
    int l
) {
    // if (left_nbr.is_cuda()){
    // 调用 CUDA 函数
    sub_suppport_affect_cuda(left_nbr, right_nbr, e_curr_rep, support, mask1, mask2, e_affect_mask, l);
    // }
    // else{
    // segment_isin2_cpu(a_rows.data_ptr<int>(), b_col_indices.data_ptr<int>(), b_row_ptr.data_ptr<int>(), output.data_ptr<int>(), output.size(0));}
}
/*******************************************************peeling_undirect**********************************************/
// 声明 CUDA 函数 e_curr, graph.rows, graph.columns,  graph.row_ptr, support, e_mask, n_mark, l, n_cut
void peeling_undirect_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor columns_g,
    const torch::Tensor row_ptr,
    const torch::Tensor edges_id_nbr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l);

// 包装函数
void peeling_undirect(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor columns_g,
    torch::Tensor row_ptr,
    torch::Tensor edges_id_nbr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr, 
    int l
) {
    // if (e_curr.is_cuda()){
    // peeling_undirect_cuda(e_curr, rows, columns, columns_g, row_ptr, edges_id_nbr, support, e_mask, n_mark, in_curr, l+1);}
    peeling_undirect_cuda(e_curr, rows, columns, columns_g, row_ptr, edges_id_nbr, support, e_mask, n_mark, in_curr, l+1);
}
/*******************************************************peeling_undirect_tile**********************************************/
// 声明 CUDA 函数 e_curr, graph.rows, graph.columns,  graph.row_ptr, support, e_mask, n_mark, l, n_cut
void peeling_undirect_tile_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor columns_g,
    const torch::Tensor row_ptr,
    const torch::Tensor edges_id_nbr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut);

void peeling_undirect_tile(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor columns_g,
    torch::Tensor row_ptr,
    torch::Tensor edges_id_nbr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr,
    int l, int n_cut
) {
    // if (e_curr.is_cuda()){
    // peeling_undirect_tile_cuda(e_curr, rows, columns, columns_g, row_ptr, edges_id_nbr, support, e_mask, n_mark, in_curr, l+1, n_cut);}
    peeling_undirect_tile_cuda(e_curr, rows, columns, columns_g, row_ptr, edges_id_nbr, support, e_mask, n_mark, in_curr, l+1, n_cut);
}

/*******************************************************peeling_direct_oo*********************************************/
void peeling_direct_oo_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l);

void peeling_direct_oo(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor row_ptr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr,
    int l
) {
    // if (e_curr.is_cuda()){
    // peeling_direct_oo_cuda(e_curr, rows, columns, row_ptr, support, e_mask, n_mark, in_curr, l+1);}
    peeling_direct_oo_cuda(e_curr, rows, columns, row_ptr, support, e_mask, n_mark, in_curr, l+1);
}

/******************peeling_direct_tile_oo(e_curr, rows, columns, row_ptr, support, e_mask, n_mark, in_curr, l, n_cut)**************/
void peeling_direct_tile_oo_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut);

void peeling_direct_tile_oo(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor row_ptr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr,
    int l, int n_cut
) {
    // if (e_curr.is_cuda()){
    // peeling_direct_tile_oo_cuda(e_curr, rows, columns, row_ptr, support, e_mask, n_mark, in_curr, l+1, n_cut);}
    peeling_direct_tile_oo_cuda(e_curr, rows, columns, row_ptr, support, e_mask, n_mark, in_curr, l+1, n_cut);
}
/*******************************************************peeling_direct_ii*********************************************/
void peeling_direct_ii_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l);

void peeling_direct_ii(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor r_edges,
    torch::Tensor re_ptr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr,
    int l
) {
    // if (e_curr.is_cuda()){
    // peeling_direct_ii_cuda(e_curr, rows, columns, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1);}
    peeling_direct_ii_cuda(e_curr, rows, columns, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1);
}

/****************** peeling_direct_tile_ii(e_curr, rows, columns, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l, n_cut)**************/
void peeling_direct_tile_ii_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut);

void peeling_direct_tile_ii(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor r_edges,
    torch::Tensor re_ptr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr,
    int l, int n_cut
) {
    // if (e_curr.is_cuda()){
    // peeling_direct_tile_ii_cuda(e_curr, rows, columns, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1, n_cut);}
    peeling_direct_tile_ii_cuda(e_curr, rows, columns, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1, n_cut);
}

/*******************************************************peeling_direct_oi*********************************************/
void peeling_direct_oi_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l);

void peeling_direct_oi(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor row_ptr,
    torch::Tensor r_edges,
    torch::Tensor re_ptr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr,
    int l
) {
    // if (e_curr.is_cuda()){
    // peeling_direct_oi_cuda(e_curr, rows, columns, row_ptr, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1);}
    peeling_direct_oi_cuda(e_curr, rows, columns, row_ptr, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1);
}

/******************peeling_direct_tile_oi(e_curr, rows, columns, row_ptr, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l, n_cut)*****************/
void peeling_direct_tile_oi_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut);

void peeling_direct_tile_oi(torch::Tensor e_curr,
    torch::Tensor rows,
    torch::Tensor columns,
    torch::Tensor row_ptr,
    torch::Tensor r_edges,
    torch::Tensor re_ptr,
    torch::Tensor support,
    torch::Tensor e_mask, 
    torch::Tensor n_mark,
    torch::Tensor in_curr,
    int l,
    int n_cut
) {
    // if (e_curr.is_cuda()){
    // peeling_direct_tile_oi_cuda(e_curr, rows, columns, row_ptr, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1, n_cut);}
    peeling_direct_tile_oi_cuda(e_curr, rows, columns, row_ptr, r_edges, re_ptr, support, e_mask, n_mark, in_curr, l+1, n_cut);
}

/*************segment_ismm********/

// 声明 CUDA 函数
void segment_isinmm_cuda(
    const torch::Tensor u_clos,
    const torch::Tensor v_clos,
    const torch::Tensor uptr,
    const torch::Tensor vptr,
    torch::Tensor u_mask,
    torch::Tensor v_mask
);

// 包装函数
void segment_isinmm(
    torch::Tensor u_clos,
    torch::Tensor v_clos,
    torch::Tensor uptr,
    torch::Tensor vptr,
    torch::Tensor u_mask,
    torch::Tensor v_mask
) {
    // 调用 CUDA 函数
    segment_isinmm_cuda(u_clos, v_clos, uptr, vptr, u_mask, v_mask);
}





//****************************python module generating*************************************************************//
PYBIND11_MODULE(trusstensor, m) {
    m.def("segment_add", &segment_add, "Segment add operation");
    m.def("segment_isin2", &segment_isin2, "Segment isin2 operation on CUDA");
    m.def("segment_isin2tile", &segment_isin2tile, "Segment_isin2tile operation on CUDA");
    m.def("sub_AllAffectedSupport", &sub_AllAffectedSupport, "Sub_AllAffectedSupport operation on CUDA");
    m.def("sub_AllAffectedSupport_not", &sub_AllAffectedSupport_not, "Sub_AllAffectedSupport_not operation on CUDA");
    m.def("sub_AllAffectedSupport_notwin", &sub_AllAffectedSupport_notwin, "Sub_AllAffectedSupport_notwin operation on CUDA");
    m.def("sub_AllAffectedSupport_tile", &sub_AllAffectedSupport_tile, "sub_AllAffectedSupport_tile operation on CUDA");
    m.def("sub_AllAffectedSupport_tilenot", &sub_AllAffectedSupport_tilenot, "sub_AllAffectedSupport_tilenot operation on CUDA");
    m.def("sub_AllAffectedSupport_tilenotwin", &sub_AllAffectedSupport_tilenotwin, "sub_AllAffectedSupport_tilenotwin operation on CUDA");
    m.def("segment_isinis_m", &segment_isinis_m, "segment_isinis_m operation on CUDA");
    m.def("sub_suppport_affect", &sub_suppport_affect, "sub_suppport_affect operation on CUDA");
    m.def("peeling_undirect_tile", &peeling_undirect_tile, "peeling_undirect_tile operation on CUDA");
    m.def("peeling_undirect", &peeling_undirect, "peeling_undirect operation on CUDA");
    m.def("peeling_direct_oo", &peeling_direct_oo, "peeling_direct_oo operation on CUDA");
    m.def("peeling_direct_tile_oo", &peeling_direct_tile_oo, "peeling_direct_tile_oo operation on CUDA");
    m.def("peeling_direct_ii", &peeling_direct_ii, "peeling_direct_ii operation on CUDA");
    m.def("peeling_direct_tile_ii", &peeling_direct_tile_ii, "peeling_direct_tile_ii operation on CUDA");
    m.def("peeling_direct_oi", &peeling_direct_oi, "peeling_direct_ii operation on CUDA");
    m.def("peeling_direct_tile_oi", &peeling_direct_tile_oi, "peeling_direct_tile_ii operation on CUDA");
    m.def("segment_isinmm", &segment_isinmm, "Segment isin operation on CUDA");
}
