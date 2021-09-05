#include "StateStrategoTest.h"

TEST_F(StateStrategoTest, StateStrategoTest_do_move)
{
   move_type move1{{1, 2}, {3, 2}};
   state.do_move(move1);
   ASSERT_EQ((state[{3, 2}]->get_role()), role_type(11, 0));
   ASSERT_EQ((state[{1, 2}]->get_role()), role_type(404, 404));
   ASSERT_EQ(
      state.get_board()->get_position_of_role(1, {11, 0})->second,
      position_type(3, 2));
   ASSERT_EQ(
      state.get_board()->get_position_of_role(0, {2, 2}),
      state.get_board()->end_inverse(0));

   move_type move2{{3, 1}, {2, 1}};
   state.do_move(move2);
   ASSERT_EQ((state[{3, 1}]->get_role()), role_type(404, 404));
   ASSERT_EQ((state[{2, 1}]->get_role()), role_type(2, 1));

   move_type move3{{1, 1}, {2, 1}};
   state.do_move(move3);
   ASSERT_EQ((state[{1, 1}]->get_role()), role_type(404, 404));
   ASSERT_EQ((state[{2, 1}]->get_role()), role_type(10, 0));

   move_type move4{{0, 2}, {3, 2}};
   state.do_move(move4);
   ASSERT_EQ((state[{3, 2}]->get_role()), role_type(11, 0));
   ASSERT_EQ((state[{0, 2}]->get_role()), role_type(404, 404));

   move_type move5{{3, 3}, {2, 3}};
   state.do_move(move5);
   ASSERT_EQ((state[{2, 3}]->get_role()), role_type(2, 2));
   ASSERT_EQ((state[{3, 3}]->get_role()), role_type(404, 404));

   move_type move6{{1, 4}, {2, 4}};
   state.do_move(move6);
   ASSERT_EQ((state[{2, 4}]->get_role()), role_type(3, 1));
   ASSERT_EQ((state[{1, 4}]->get_role()), role_type(404, 404));
}
