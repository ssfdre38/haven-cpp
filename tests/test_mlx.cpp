#include <iostream>
#include "mlx/mlx.h"

int main() {
    std::cout << "[Test MLX] Initializing MLX on Windows...\n" << std::flush;
    auto a = mlx::core::array({1.0f, 2.0f, 3.0f, 4.0f});
    auto b = mlx::core::array({10.0f, 20.0f, 30.0f, 40.0f});
    auto c = mlx::core::add(a, b);
    mlx::core::eval(c);
    std::cout << "[Test MLX] a + b = " << c << "\n" << std::flush;
    std::cout << "[Test MLX] MLX running successfully on Windows!\n" << std::flush;
    return 0;
}
