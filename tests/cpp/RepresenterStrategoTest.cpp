
#include "RepresenterStrategoTest.h"

template < class MoveContainer, class MaskContainer >
void fill_mask(
   const MoveContainer moves,
   const RepresenterStratego &action_rep,
   const state_type &state,
   MaskContainer &mask)
{
   for(const auto &move : moves) {
      auto actions_for_move = action_rep.get_actions_by_role(
         state[move[0]]->get_role());
      auto move_effect = move[1] - move[0];
      for(const auto &action : actions_for_move) {
         if(move_effect == action.get_effect()) {
            mask[action.get_index()] = 1;
            break;
         }
      }
   }
}

TEST_F(RepresenterStrategoTest, RepresenterStrategoTest_action_mask_Test)
{
   RepresenterStratego action_rep(5);

   auto board_start = std::make_shared<typename state_type::board_type>(
      std::array< size_t, 2 >{5, 5}, setup_start_0, setup_start_1);
   auto state_start = state_type(board_start, 17);

   std::vector< unsigned int > expect_action_mask_0_start(64, 0);
   std::vector< unsigned int > expect_action_mask_1_start(64, 0);

   fill_mask(
      logic_type::get_legal_moves_(*state_start.get_board(), 0),
      action_rep,
      state_start,
      expect_action_mask_0_start);
   fill_mask(
      logic_type::get_legal_moves_(*state_start.get_board(), 1),
      action_rep,
      state_start,
      expect_action_mask_1_start);

   auto action_mask_start_0 = action_rep.get_action_mask_(*board_start, 0);
   auto action_mask_start_1 = action_rep.get_action_mask_(*board_start, 1);

   EXPECT_EQ(action_mask_start_0, expect_action_mask_0_start);
   EXPECT_EQ(action_mask_start_1, expect_action_mask_1_start);

   auto board_mid = std::make_shared<typename state_type::board_type>(
      std::array< size_t, 2 >{5, 5}, setup_mid_0, setup_mid_1);
   auto state_mid = state_type(board_mid, 17);

   std::vector< unsigned int > expect_action_mask_0_mid(64, 0);
   std::vector< unsigned int > expect_action_mask_1_mid(64, 0);

   fill_mask(
      logic_type::get_legal_moves_(*state_mid.get_board(), 0),
      action_rep,
      state_mid,
      expect_action_mask_0_mid);
   fill_mask(
      logic_type::get_legal_moves_(*state_mid.get_board(), 1),
      action_rep,
      state_mid,
      expect_action_mask_1_mid);

   auto action_mask_mid_0 = action_rep.get_action_mask_(
      *state_mid.get_board(), 0);
   auto action_mask_mid_1 = action_rep.get_action_mask_(
      *state_mid.get_board(), 1);

   EXPECT_EQ(action_mask_mid_0, expect_action_mask_0_mid);
   EXPECT_EQ(action_mask_mid_1, expect_action_mask_1_mid);
}