#include "GameStratego.h"

GameStratego::GameStratego(
   const std::array< size_t, 2 > &shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1,
   const sptr< Agent< state_type > > &ag0,
   const sptr< Agent< state_type > > &ag1)
    : base_type(state_type(shape, setup_0, setup_1), ag0, ag1)
{
}

GameStratego::GameStratego(
   const std::array< size_t, 2 > &shape,
   const std::map< position_type, role_type > &setup_0,
   const std::map< position_type, role_type > &setup_1,
   const sptr< Agent< state_type > > &ag0,
   const sptr< Agent< state_type > > &ag1)
    : base_type(state_type(shape, setup_0, setup_1), ag0, ag1)
{
}

GameStratego::GameStratego(
   size_t shape,
   const std::map< position_type, int > &setup_0,
   const std::map< position_type, int > &setup_1,
   const sptr< Agent< state_type > > &ag0,
   const sptr< Agent< state_type > > &ag1)
    : base_type(state_type({shape, shape}, setup_0, setup_1), ag0, ag1)
{
}

GameStratego::GameStratego(
   size_t shape,
   const std::map< position_type, role_type > &setup_0,
   const std::map< position_type, role_type > &setup_1,
   const sptr< Agent< state_type > > &ag0,
   const sptr< Agent< state_type > > &ag1)
    : base_type(state_type({shape, shape}, setup_0, setup_1), ag0, ag1)
{
}

std::map< GameStratego::position_type, GameStratego::sptr_piece_type > GameStratego::draw_setup_(
   int team)
{
   int shape = m_game_state.get_board()->get_shape()[0];
   auto avail_types = LogicStratego< board_type >::get_available_types(shape);

   std::vector< position_type > poss_pos = LogicStratego< board_type >::get_start_positions(
      shape, team);

   std::map< position_type, sptr_piece_type > setup_out;

   std::random_device rd;
   std::mt19937 rng(rd());
   std::shuffle(poss_pos.begin(), poss_pos.end(), rng);
   std::shuffle(avail_types.begin(), avail_types.end(), rng);

   auto counter = utils::counter(avail_types);

   while(! poss_pos.empty()) {
      auto &pos = poss_pos.back();
      auto &type = avail_types.back();

      setup_out[pos] = std::make_shared< piece_type >(
         pos, typename piece_type::role_type(type, --counter[type]), team);

      poss_pos.pop_back();
      avail_types.pop_back();
   }
   return setup_out;
}

int GameStratego::check_terminal()
{
   auto state = get_state();
   if(auto dead_pieces = state.get_graveyard(0);
      std::find(dead_pieces.begin(), dead_pieces.end(), role_type{0, 0}) != dead_pieces.end()) {
      // flag of player 0 has been captured (killed), therefore player 0 lost
      state.set_terminality(WIN_1_FLAG);
   } else if(dead_pieces = state.get_graveyard(1);
             std::find(dead_pieces.begin(), dead_pieces.end(), role_type{0, 0})
             != dead_pieces.end()) {
      // flag of player 1 has been captured (killed), therefore player 1 lost
      state.set_terminality(WIN_0_FLAG);
   }

   // committing draw rules here

   // Rule 1: If either player has no moves left.
   else if(not LogicStratego< board_type >::has_legal_moves_(*state.get_board(), 0)) {
      state.set_terminality(TIE);
   } else if(not LogicStratego< board_type >::has_legal_moves_(*state.get_board(), 1)) {
      state.set_terminality(TIE);
   }

   // Rule 2: If the moves of both players have been repeated too often times.
   else if(auto [hist, nr_rounds] = std::pair(state.get_history(), MAX_REPEATED_ROUNDS * 2 * 2);
           hist.size() >= nr_rounds) {
      // times 2, because we need to look at every second turn to possibly do a repeated move
      // times 2 again, because we have two teams.

      auto last_turn = hist.end();
      std::array< std::vector< move_type >, 2 > turns;
      for(int i = 0; i < nr_rounds; i++) {
         auto &&turn = *(last_turn - i);
         turns[std::get< 0 >(turn)].emplace_back(std::get< 1 >(turn));
      }
      auto repetition_lambda = [&](int team) {
         auto &ts = turns[team];
         for(int i = 0; i < ts.size() - 2; i++) {
            if(ts[i] != ts[i + 2]) {
               return false;
            }
         }
         return true;
      };

      if(repetition_lambda(0) && repetition_lambda(1))
         // players simply repeated their last 3 moves -> draw
         state.set_terminality(TIE);
   }
   // Rule 3: If no fight has happened for 50 rounds in a row.
   else if(m_rounds_without_fight > 49) {
      m_terminal = 0;
   }
   m_terminal_checked = true;
   return m_terminal;
}