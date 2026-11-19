#include <stdio.h>
#include "mlx/c/mlx.h"

int main() {
    printf("[MLX-C Test] Calling MLX on Windows via mlxc C-ABI...\n");
    mlx_array a = mlx_array_new_float(42.0f);
    printf("[MLX-C Test] Created array a successfully!\n");
    mlx_array_free(a);
    printf("[MLX-C Test] Freed array a successfully! MLX is operational.\n");
    return 0;
}
