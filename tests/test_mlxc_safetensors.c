#include <stdio.h>
#include "mlx/c/mlx.h"

int main() {
    printf("[MLX-C Safetensors Test] Loading safetensors via mlxc on Windows...\n");
    const char* file = "C:/Users/admin/gemma4-turbo-family/shieldgemma-2b/model-00001-of-00002.safetensors";
    
    mlx_map_string_to_array tensors = mlx_map_string_to_array_new();
    mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
    
    mlx_device dev = mlx_device_new_type(MLX_CPU, 0);
    mlx_stream s = mlx_stream_new_device(dev);
    
    int ret = mlx_load_safetensors(&tensors, &metadata, file, s);
    printf("[MLX-C Safetensors Test] mlx_load_safetensors returned: %d\n", ret);
    
    if (ret == 0) {
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(tensors);
        const char* key = NULL;
        mlx_array val = mlx_array_new();
        int count = 0;
        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 && count < 10) {
            printf("  [Tensor %d] %s\n", count++, key);
        }
        mlx_map_string_to_array_iterator_free(it);
    }
    
    mlx_map_string_to_array_free(tensors);
    mlx_map_string_to_string_free(metadata);
    mlx_stream_free(s);
    mlx_device_free(dev);
    printf("[MLX-C Safetensors Test] Done!\n");
    return 0;
}
