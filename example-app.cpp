#include <torch/script.h>
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-image>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  assert(module != nullptr);
  std::cout << "load model ok\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::rand({256, 3, 224, 224}));

  // evalute time
  double t = (double)cv::getTickCount();
  module->forward(inputs).toTensor();
  t = (double)cv::getTickCount() - t;
  printf("execution time = %gs\n", t / cv::getTickFrequency());
  inputs.pop_back();

  // load image with opencv
  cv::Mat image;
  image = cv::imread( argv[2], 1 );
  resize(image, image, cv::Size(224,224));
  cv::Mat img_float;
  image.convertTo(img_float, CV_32F, 1.0/255);
  auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, 3, 224, 224});
  auto img_var = torch::autograd::make_variable(img_tensor, false);
  //std::cout << img_var << std::endl;
  inputs.push_back(img_var);
  
  // Execute the model and turn its output into a tensor.
  auto output = module->forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}

