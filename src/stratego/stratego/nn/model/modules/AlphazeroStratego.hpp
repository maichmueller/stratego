#pragma once

#include <aze/nn.h>
#include <aze/utils.h>
#include <torch/torch.h>

class StrategoAlphaZero: public AlphaZeroBase {
   int D_in;
   sptr< Convolutional > convo_layers;
   sptr< FullyConnected > linear_layers;
   torch::nn::Linear pi_act_layer;
   torch::nn::Linear v_act_layer;

  public:
   StrategoAlphaZero(
      int D_in,
      int D_out,
      int nr_lin_layers,
      int start_exponent,
      int channels,
      const std::vector< unsigned int > &filter_sizes,
      const std::vector< unsigned int > &kernel_sizes_vec,
      const std::vector< bool > &maxpool_used_vec,
      const std::vector< float > &dropout_probs,
      const torch::nn::Functional &activation_function = torch::nn::Functional(torch::relu));

   std::tuple< torch::Tensor, torch::Tensor > forward(const torch::Tensor &input) override;
};

StrategoAlphaZero::StrategoAlphaZero(
   int D_in,
   int D_out,
   int nr_lin_layers,
   int start_exponent,
   int channels,
   const std::vector< unsigned int > &filter_sizes,
   const std::vector< unsigned int > &kernel_sizes_vec,
   const std::vector< bool > &maxpool_used_vec,
   const std::vector< float > &dropout_probs,
   const torch::nn::Functional &activation_function)
    : D_in(D_in),
      convo_layers(std::make_shared< Convolutional >(
         channels,
         filter_sizes,
         kernel_sizes_vec,
         maxpool_used_vec,
         dropout_probs,
         activation_function)),
      pi_act_layer(nullptr),
      v_act_layer(nullptr)
{
   if(nr_lin_layers < 2) {
      throw std::invalid_argument("Less than 2 linear layers requested. Aborting.");
   }

   unsigned int hidden_nodes = 2u << static_cast< unsigned int >(start_exponent - 1);
   unsigned int substitute_d_out = hidden_nodes
                                   / (2u << static_cast< unsigned int >(nr_lin_layers - 4));

   linear_layers = std::make_shared< FullyConnected >(
      D_in, substitute_d_out, nr_lin_layers - 1, start_exponent, activation_function);

   pi_act_layer = torch::nn::Linear(torch::nn::LinearOptions(substitute_d_out, D_out));
   v_act_layer = torch::nn::Linear(torch::nn::LinearOptions(substitute_d_out, 1));

   convo_layers = register_module("Convolutional Block", convo_layers);
   linear_layers = register_module("FullyConnected Block", linear_layers);
   pi_act_layer = register_module("Policy Generating Layer", pi_act_layer);
   v_act_layer = register_module("Value Generating Layer", v_act_layer);
}

std::tuple< torch::Tensor, torch::Tensor > StrategoAlphaZero::forward(const torch::Tensor &input)
{
   torch::Tensor features = convo_layers->forward(input);
   features = linear_layers->forward(features.view({-1, D_in}));

   torch::Tensor pi = torch::log_softmax(pi_act_layer->forward(features), 1);
   torch::Tensor v = torch::tanh(v_act_layer->forward(features));

   return std::make_tuple(pi, v);
}