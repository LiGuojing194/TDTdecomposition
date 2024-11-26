#include <torch/extension.h>
#include <omp.h>

void segment_add_cpu(const int *col_indices, const int *col_ptr, int *output, const int numRows) {
    #pragma omp parallel for
    for (int row = 0; row < numRows; row++) {
        // int num_threads = omp_get_num_threads();
        // int thread_id = omp_get_thread_num();
        // if (thread_id == 0) {
        //     std::cout << "Total threads: " << num_threads << std::endl;
        // }
        int start = col_ptr[row];
        int end = col_ptr[row + 1];
        if (start < end) {
            int sum = col_indices[start];
            start += 1;
            while (start < end) {
                sum += col_indices[start];
                start += 1;
            }
            output[row] = sum;
        }
    }
}

void segment_isin2_cpu(const int *a_rows, const int *b_col_indices,
                            const int *b_row_ptr,
                            int *output, const int numRows) {
    #pragma omp parallel for
    for (int row = 0; row < numRows; row++) {
        int col = b_col_indices[row];
        int idxA = b_row_ptr[col];
        int endA = b_row_ptr[col + 1];
        int a_row = a_rows[row];
        int idxB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row + 1];
        int count = 0;

        while (idxA < endA && idxB < endB) {
            int colA = b_col_indices[idxA];
            int colB = b_col_indices[idxB];
            if (colA == colB) {
                #pragma omp atomic
                output[idxA]++;
                #pragma omp atomic
                output[idxB]++;
                count++;
                idxA++;
                idxB++;
            } else if (colA < colB) {
                idxA++;
            } else {
                idxB++;
            }
        }
        #pragma omp atomic
        output[row] += count;
    }
}

void segment_isin2tile_cpu(const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut, 
    int *output, const int numRows
) {
    #pragma omp parallel for
    for (int row = 0; row < numRows; row++) {
        int i = row / n_cut;
        int j = row % n_cut;
        int col = b_col_indices[i]*n_cut + j;
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[i]*n_cut + j;
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;

        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                #pragma omp atomic
                output[startA]++;
                #pragma omp atomic
                output[startB]++;
                count++;
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        #pragma omp atomic
        output[i] += count;
    }
}

void AllAffectedSupport_cpu(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows
) {
    #pragma omp parallel for
    for (int row = 0; row < numRows; row++) {
        int e = e_affect[row];
        int col = b_col_indices[e];
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e];
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && (mark[e] || mark[startA] || mark[startB])){
                    if (output[startA]>l){
                        #pragma omp atomic
                        output[startA]--;
                        // atomicSub(&output[startA], 1);
                        n_mark[startA] = true;
                    }
                    if (output[startB]>l){
                        #pragma omp atomic
                        output[startB]--;
                        // atomicSub(&output[startB], 1);
                        n_mark[startB] = true;
                    }
                    if (!mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
            #pragma omp atomic
            output[e] -= count;
            // atomicSub(&output[e], count);
            n_mark[e] = true;
        }
    }
}

void AllAffectedSupport_tile_cpu(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut,  const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows
) {
    #pragma omp parallel for
    for (int row = 0; row < numRows; row++) {
        int i = row / n_cut;
        int j = row  % n_cut;
        int e = e_affect[i];
        int col = b_col_indices[e]*n_cut + j;
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e]*n_cut + j;
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && (mark[e] || mark[startA] || mark[startB])){
                    if (output[startA]>l){
                        #pragma omp atomic
                        output[startA]--;
                        // atomicSub(&output[startA], 1);
                        n_mark[startA] = true;
                    }
                    if (output[startB]>l){
                        #pragma omp atomic
                        output[startB]--;
                        // atomicSub(&output[startB], 1);
                        n_mark[startB] = true;
                    }
                    if (!mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
                #pragma omp atomic
                output[e] -= count;
                // atomicSub(&output[e], count);
                n_mark[e] = true;
        }
    }
}


