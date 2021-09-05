//
// Created by michael on 30.05.19.
//

#ifndef STRATEGO_CPP_STATEREPRESENTATION_H
#define STRATEGO_CPP_STATEREPRESENTATION_H

#include <memory>
#include <vector>

#include "board/Piece.h"
#include "game/GameUtilsStratego.h"
#include "torch/torch.h"
#include "utils/torch_utils.h"

namespace StateRepresentation {

inline Position pos_ident(int& len, const Position& pos)
{
   return pos;
}
inline Position pos_invert(int& len, const Position& pos)
{
   Position p = {len - pos[0] - 1, len - pos[1] - 1};
   return p;
}
inline int team_ident(int team)
{
   return team;
}
inline int team_invert(int team)
{
   return 1 - team;
}

template < typename Piece >
inline bool check_condition(
   const sptr< Piece >& piece,
   int team,
   int type,
   int version,
   bool hidden,
   bool flip_teams = false)
{
   // if we flip the teams, we want pieces of m_team 1 to appear as m_team 0
   // and vice versa
   int team_piece = flip_teams ? 1 - piece->get_team() : piece->get_team();

   if(team == 0) {
      if(! hidden) {
         // if it is about m_team 0, the 'hidden' status is unimportant
         // (since the alpha zero agent always plays from the perspective
         // of player 0, therefore it can see all its own pieces)
         bool eq_team = team_piece == team;
         bool eq_type = piece->get_type() == type;
         bool eq_vers = piece->get_version() == version;
         return eq_team && eq_type && eq_vers;
      } else {
         // 'hidden' is only important for the single condition that
         // specifically checks for this property (information about own pieces
         // visible or not).
         bool eq_team = team_piece == team;
         bool hide = piece->get_flag_hidden() == hidden;
         return eq_team && hide;
      }
   } else if(team == 1) {
      // for m_team 1 we only get the info about type and version if it isn't
      // hidden otherwise it will fall into the 'hidden' layer
      if(! hidden) {
         if(piece->get_flag_hidden())
            return false;
         else {
            bool eq_team = team_piece == team;
            bool eq_type = piece->get_type() == type;
            bool eq_vers = piece->get_version() == version;
            return eq_team && eq_type && eq_vers;
         }
      } else {
         bool eq_team = team_piece == team;
         bool hide = piece->get_flag_hidden() == hidden;
         return eq_team && hide;
      }
   } else {
      // only the obstacle should reach here
      return team_piece == team;
   }
}

template < typename Board >
inline torch::Tensor b2s_cond_check(
   const Board& board,
   const std::vector< std::tuple< int, int, int, bool > >& conditions,
   int player = 0)
{
   /**
    * We are trying to build a state representation of a Stratego board.
    * To this end, 'conditions' are evaluated for each
    * piece on the board. These 'conditions' are checked in sequence.
    * Each condition receives its own layer with 0's everywhere, except
    * for where the specific condition was true, which holds a 1.
    * |==========================================================================|
    * |              In short: x conditions -> x binary layers | | (one for each
    *condition)                      |
    * |==========================================================================|
    *
    * Parameters
    * ----------
    * @param board, the board whose representation we want
    * @param conditions, std::vector of tuples for the conditions,
    *      on which to check the board
    * @param player, int deciding which player's representation we're seeking
    *
    * Returns
    * -------
    * @return tensor of 0's and 1's on the positions for which the relevant
    *condition was true (1) or wrong (0)
    **/

   std::function< Position(int&, Position&) > canonize_pos = &pos_ident;
   std::function< int(int) > canonize_team = &team_ident;

   int board_len = board.get_board_len();
   int state_dim = conditions.size();
   bool flip_teams = static_cast< bool >(player);

   if(flip_teams) {
      canonize_pos = &pos_invert;
      canonize_team = &team_invert;
   }

   auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(GLOBAL_DEVICE::get_device())
                     .requires_grad(true);
   // the dimensions here are as follows:
   // 1 = batch_size (in this case obvciously only 1)
   // state_dim = dimension of the state rep, i.e. how many m_layers of the
   // conditions m_shape = first board dimension m_shape = second board
   // dimension
   torch::Tensor board_state_rep = torch::zeros({1, state_dim, board_len, board_len}, options);

   //        auto board_state_access = board_state_rep.accessor<float, 4> ();
   for(const auto& pos_piece : board) {
      Position pos = pos_piece.first;
      pos = canonize_pos(board_len, pos);
      auto piece = pos_piece.second;
      if(! piece->is_null()) {
         for(auto&& [i, cond_it] = std::make_tuple(0, conditions.begin());
             cond_it != conditions.end();
             ++i, ++cond_it) {
            // unpack the condition
            auto [team, type, vers, hidden] = *cond_it;
            // write the result of the condition check to the tensor
            board_state_rep[0][i][pos[0]][pos[1]] = check_condition(
               piece, team, type, vers, hidden, flip_teams);
         }
      }
   }
   // send the tensor to the global device for
   // working with the GPU if possible
   board_state_rep.to(GLOBAL_DEVICE::get_device());

   return board_state_rep;
}

using cond_type = std::tuple< int, int, int, bool >;

inline std::vector< cond_type > create_conditions(
   const std::map< int, unsigned int >& type_counter, int own_team)
{
   std::vector< std::tuple< int, int, int, bool > > conditions(0);

   // own m_team 0
   // [flag, 1, 2, 3, 4, ..., 10, bombs] UNHIDDEN
   for(const auto& entry : type_counter) {
      int type = entry.first;
      for(int version = 0; version < entry.second; ++version) {
         conditions.emplace_back(std::make_tuple(own_team, type, version, false));
      }
   }
   // [all own pieces] HIDDEN
   // Note: type and version info are unused
   // in the check in this case (thus -1)
   conditions.emplace_back(std::make_tuple(own_team, -1, -1, true));

   // enemy m_team 1
   // [flag, 1, 2, 3, 4, ..., 10, bombs] UNHIDDEN
   for(const auto& entry : type_counter) {
      int type = entry.first;
      for(int version = 0; version < entry.second; ++version) {
         conditions.emplace_back(std::make_tuple(1 - own_team, type, version, false));
      }
   }
   // [all enemy pieces] HIDDEN
   // Note: type and version info are unused
   // in the check in this case (thus -1)
   conditions.emplace_back(std::make_tuple(1 - own_team, -1, -1, true));

   return conditions;
}

static std::vector< cond_type > state_torch_conv_conditions_0;
static std::vector< cond_type > state_torch_conv_conditions_1;

static bool conditions_set = false;

inline void set_state_rep_conditions(int game_len)
{
   if(conditions_set)
      return;

   auto t_count = utils::counter(GameDeclarations::get_available_types(game_len));

   if(game_len == 5) {
      state_torch_conv_conditions_0 = create_conditions(t_count, 0);
      state_torch_conv_conditions_1 = create_conditions(t_count, 1);
   } else if(game_len == 7) {
      state_torch_conv_conditions_0 = create_conditions(t_count, 0);
      state_torch_conv_conditions_1 = create_conditions(t_count, 1);
   } else if(game_len == 10) {
      state_torch_conv_conditions_1 = create_conditions(t_count, 1);
      state_torch_conv_conditions_0 = create_conditions(t_count, 0);
   }
}

};  // namespace StateRepresentation

#endif  // STRATEGO_CPP_STATEREPRESENTATION_H
