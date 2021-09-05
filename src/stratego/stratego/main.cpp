#include <aze/aze.h>
#include <aze/nn/training/Coach.h>
#include "nn/model/modules/AlphazeroStratego.h"

#include <array>
#include <iostream>
#include <iostream>
#include <memory>

#include "board/PieceStratego.h"
#include "game/GameStratego.h"
#include "game/StateStratego.h"
#include "nn/representation/RepresenterStratego.h"
#include "torch/torch.h"

int main()
{
   std::cout << "Cuda status: ";
   if(torch::cuda::is_available()) {
      std::cout << "Available! Using CUDA...\n";
   } else {
      std::cout << "Unvailable! Using CPU...\n";
   }
   size_t board_size = 5;

   //
   // create the action representer
   //
   auto action_rep_sptr = std::make_shared< RepresenterStratego >(5);

   //
   // build the neural network
   //
   std::vector< unsigned int > filters{128, 128, 128, 128};
   auto alphazero_net_ptr = std::make_shared< StrategoAlphaZero >(
      board_size * board_size * filters.front(),
      action_rep_sptr->get_actions().size(),
      5,
      10,
      action_rep_sptr->get_conditions().size(),
      filters,
      std::vector< unsigned int >{3, 3, 3, 3},
      std::vector< bool >{false, false, false, false},
      std::vector< float >{0.0, 0.0, 0.0, 0.0});
//   alphazero_net_ptr->to(GLOBAL_DEVICE::get_device());
   auto network_0 = std::make_shared< NetworkWrapper >(alphazero_net_ptr);
   network_0->to(GLOBAL_DEVICE::get_device());
   auto network_1 = std::make_shared< NetworkWrapper >(*network_0);

   //
   // build the agents to train.
   //
   auto agent_0 = std::make_shared<
      AlphaZeroAgent< StateStratego, RepresenterStratego > >(
      0, network_0, action_rep_sptr);
   auto agent_1 = std::make_shared<
      AlphaZeroAgent< StateStratego, RepresenterStratego > >(
      1, network_1, action_rep_sptr);
   //    auto agent_0 = std::make_shared<RandomAgent<StateStratego>>(
   //            0
   //    );
   //    auto agent_1 = std::make_shared<RandomAgent<StateStratego>>(
   //            1
   //    );

   //
   // setup the game
   //
   std::map< BoardStratego::position_type, BoardStratego::role_type > setup0;
   std::map< BoardStratego::position_type, BoardStratego::role_type > setup1;

   setup0[{0, 0}] = {0, 0};
   setup0[{0, 1}] = {1, 0};
   setup0[{0, 2}] = {2, 0};
   setup0[{0, 3}] = {2, 1};
   setup0[{0, 4}] = {3, 0};
   setup0[{1, 0}] = {11, 0};
   setup0[{1, 1}] = {10, 0};
   setup0[{1, 2}] = {2, 2};
   setup0[{1, 3}] = {11, 1};
   setup0[{1, 4}] = {3, 1};
   setup1[{3, 0}] = {2, 0};
   setup1[{3, 1}] = {2, 1};
   setup1[{3, 2}] = {11, 0};
   setup1[{3, 3}] = {2, 2};
   setup1[{3, 4}] = {10, 0};
   setup1[{4, 0}] = {3, 0};
   setup1[{4, 1}] = {1, 0};
   setup1[{4, 2}] = {11, 1};
   setup1[{4, 3}] = {3, 1};
   setup1[{4, 4}] = {0, 0};

   auto g = GameStratego(
      std::array< size_t, 2 >{5, 5}, setup0, setup1, agent_0, agent_1);
   auto game = std::make_shared< GameStratego >(
      std::array< size_t, 2 >{5, 5}, setup0, setup1, agent_0, agent_1);


   //
   // run/train on the game
   //

   //    game->run_game(false);
   Coach coach(game, network_0, "./checkpoints", 100, 1, 100);
   coach.teach(*action_rep_sptr, false, false, false, false);

   return 0;
}