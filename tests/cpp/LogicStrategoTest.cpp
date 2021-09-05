

#include "LogicStrategoTest.h"

#include <algorithm>

#include "unordered_set"

TEST_F(LogicStrategoTest, LogicStrategoTest_poss_moves_Test)
{
   auto board_start = std::make_shared<typename state_type::board_type>(
      std::array< size_t, 2 >{5, 5}, setup_start_0, setup_start_1);
   auto state_start = state_type(board_start, 17);
   auto poss_moves_0 = logic_type::get_legal_moves_(
      *state_start.get_board(), 0);
   auto poss_moves_1 = logic_type::get_legal_moves_(
      *state_start.get_board(), 1);

   std::unordered_set< move_type > real_poss_moves_0{
      {{1, 1}, {2, 1}}, {{1, 4}, {2, 4}}};
   std::unordered_set< move_type > real_poss_moves_1{
      {{3, 0}, {1, 0}},
      {{3, 0}, {2, 0}},
      {{3, 1}, {1, 1}},
      {{3, 1}, {2, 1}},
      {{3, 3}, {1, 3}},
      {{3, 3}, {2, 3}},
   };
   std::unordered_set< move_type > poss_moves_set_0(
      poss_moves_0.begin(), poss_moves_0.end());
   std::unordered_set< move_type > poss_moves_set_1(
      poss_moves_1.begin(), poss_moves_1.end());
   //
   //    std::cout << state_start.get_board()->print_board(false, false);
   //    std::cout << "player 0, calculated moves:\n";
   //    for(auto move : poss_moves_set_0) {
   //        std::cout << move.to_string() << std::endl;
   //    }
   //    std::cout << "player 1, calculated moves:\n";
   //    for(auto move : poss_moves_set_1) {
   //        std::cout << move.to_string() << std::endl;
   //    }

   EXPECT_EQ(poss_moves_set_0, real_poss_moves_0);
   EXPECT_EQ(poss_moves_set_1, real_poss_moves_1);

   auto board_mid = std::make_shared<typename state_type::board_type>(
      std::array< size_t, 2 >{5, 5}, setup_mid_0, setup_mid_1);
   auto state_mid = state_type(board_mid, 17);
   auto poss_moves_mid_0 = logic_type::get_legal_moves_(
      *state_mid.get_board(), 0);
   auto poss_moves_mid_1 = logic_type::get_legal_moves_(
      *state_mid.get_board(), 1);

   std::unordered_set< move_type > real_poss_moves_mid_0{{{1, 1}, {1, 2}},
                                                         {{1, 1}, {2, 1}},
                                                         {{1, 1}, {0, 1}},
                                                         {{0, 2}, {0, 1}},
                                                         {{0, 2}, {0, 3}},
                                                         {{0, 2}, {1, 2}},
                                                         {{0, 4}, {0, 3}},
                                                         {{0, 4}, {1, 4}},
                                                         {{2, 4}, {2, 3}},
                                                         {{2, 4}, {3, 4}},
                                                         {{2, 4}, {1, 4}}};
   std::unordered_set< move_type > real_poss_moves_mid_1{
      {{3, 0}, {1, 0}},
      {{3, 0}, {2, 0}},
      {{3, 1}, {1, 1}},
      {{3, 1}, {2, 1}},
      {{3, 1}, {3, 3}},
      {{3, 1}, {3, 2}},
      {{2, 3}, {1, 3}},
      {{2, 3}, {0, 3}},
      {{2, 3}, {3, 3}},
      {{2, 3}, {4, 3}},
      {{2, 3}, {2, 4}},
   };
   std::unordered_set< move_type > poss_moves_set_mid_0(
      poss_moves_mid_0.begin(), poss_moves_mid_0.end());
   std::unordered_set< move_type > poss_moves_set_mid_1(
      poss_moves_mid_1.begin(), poss_moves_mid_1.end());

   //    std::cout << state_mid.get_board()->print_board(false, false);
   //    std::cout << "player 0, calculated moves:\n";
   //    for(auto move : poss_moves_set_mid_0) {
   //        std::cout << move.to_string() << std::endl;
   //    }
   //    std::cout << "player 1, calculated moves:\n";
   //    for(auto move : poss_moves_set_mid_1) {
   //        std::cout << move.to_string() << std::endl;
   //    }

   EXPECT_EQ(poss_moves_set_mid_0, real_poss_moves_mid_0);
   EXPECT_EQ(poss_moves_set_mid_1, real_poss_moves_mid_1);
}