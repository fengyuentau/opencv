#include "tim/vx/context.h"
#include "tim/vx/tensor.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/conv2d.h"

int main(int /*argc*/, char** /*argv*/)
{
    auto context = tim::vx::Context::Create();
    return 0;
}