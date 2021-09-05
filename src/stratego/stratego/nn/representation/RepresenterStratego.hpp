
#pragma once

#include <aze/nn.h>
#include <torch/torch.h>

#include "game/StateStratego.h"

class RepresenterStratego: public RepresenterBase< StateStratego, RepresenterStratego > {
  public:
   using state_type = StateStratego;
   using position_type = state_type::position_type;
   using role_type = state_type::piece_type::role_type;
   using move_type = state_type::move_type;
   using piece_type = state_type::piece_type;
   using board_type = state_type::board_type;
   using action_type = Action< position_type, role_type >;
   using condition_container = std::vector<
      std::tuple< typename BoardStratego::piece_type::role_type, int, bool > >;
   static_assert(
      std::is_same_v< typename move_type::position_type, typename board_type::position_type >);

   explicit RepresenterStratego(size_t shape)
       : RepresenterStratego(_build_actions(shape), _build_conditions(shape))
   {
   }

   RepresenterStratego(size_t shape, const condition_container &conditions)
       : RepresenterStratego(_build_actions(shape), conditions)
   {
   }

   torch::Tensor state_representation_(const state_type &state, int player) const
   {
      return state_representation_(state, player, m_conditions);
   }

   /**
    * Method to convert a given state to a torch tensor representation.
    *
    * This method will not accumulate gradients in its representation as this
    * will otherwise blow up memory usage in the MCTS class.
    *
    * @tparam condition_type the conditions type we want to handle. At the current moment this only
    * handles the default type. Others would lead to a compile time error.
    * @param state the state to convert.
    * @param player the current active player.
    * @param conditions the vector of conditions. For each condition the first size of the torch
    * tensor is incremented.
    * @return torch tensor representing the state.
    */
   template < typename condition_type = std::tuple< role_type, int, bool > >
   torch::Tensor state_representation_(
      const state_type &state, int player, std::vector< condition_type > conditions) const;

   [[nodiscard]] const auto &get_actions_() const { return m_actions; }

   [[nodiscard]] const auto &get_role_to_actions_map() const { return m_role_to_actions_map; }

   [[nodiscard]] const auto &get_conditions() const { return m_conditions; }

   [[nodiscard]] const std::vector< action_type > &get_actions_by_role(const role_type &role) const
   {
      return m_role_to_actions_map.at(role);
   }

   template < typename Board >
   std::vector< unsigned int > get_action_mask_(const Board &board, int player);

   template < typename Board >
   static std::vector< unsigned int > get_action_mask_(
      const std::vector< action_type > &actions, const Board &board, int player);

  private:
   // Delegator constructor to initialize both const fields actions and
   // role_to_actions_map.
   RepresenterStratego(
      std::tuple<
         std::vector< action_type >,
         std::unordered_map< role_type, std::vector< action_type > > > &&actions_map,
      const condition_container &conditions)
       : m_actions(std::move(std::get< 0 >(actions_map))),
         m_role_to_actions_map(std::move(std::get< 1 >(actions_map))),
         m_conditions(conditions)
   {
   }

   static std::tuple<
      std::vector< action_type >,
      std::unordered_map< role_type, std::vector< action_type > > >
   _build_actions(size_t shape);

   static condition_container _build_conditions(size_t shape);

   template < typename Piece >
   inline bool _check_condition(
      const sptr< Piece > &piece,
      const role_type &role,
      int team,
      bool hidden,
      bool flip_teams = false) const;

   const std::vector< action_type > m_actions;
   const std::unordered_map< role_type, std::vector< action_type > > m_role_to_actions_map;
   const condition_container m_conditions;
};

template < typename condition_type >
torch::Tensor RepresenterStratego::state_representation_(
   const state_type &state, int player, std::vector< condition_type > conditions) const
{
   auto board = state.get_board();
   auto shape = board->get_shape();
   auto starts = board->get_starts();
   int state_dim = conditions.size();
   bool flip_teams = static_cast< bool >(player);

   std::function< position_type(position_type &) > canonize_pos = [&](position_type &pos) {
      return pos;
   };
   std::function< int(int) > canonize_team = [](int team) { return team; };

   if(flip_teams) {
      canonize_pos = [&](position_type &pos) { return pos.invert(starts, shape); };
      canonize_team = [](int team) { return 1 - team; };
   }

   auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(GLOBAL_DEVICE::get_device())
                     .requires_grad(false);
   // the dimensions here are as follows:
   // state_dim = dimension of the state rep, i.e. how many layers of the
   // conditions shape[0] = first board dimension shape[1] = second board
   // dimension
   torch::Tensor board_state_rep = torch::zeros(
      {
         1,  // batch size
         state_dim,  // nr conditions
         static_cast< long >(shape[0]),  // board shape x
         static_cast< long >(shape[1])  // board shape y
      },
      options);

   for(const auto &pos_piece : *board) {
      Position pos = pos_piece.first;
      pos = canonize_pos(pos);
      auto piece = pos_piece.second;
      if(! piece->is_null()) {
         for(auto &&[i, cond_it] = std::make_tuple(0, conditions.begin());
             cond_it != conditions.end();
             ++i, ++cond_it) {
            // unpack the condition
            auto [role, team, hidden] = *cond_it;
            // write the result of the condition check to the tensor
            board_state_rep[0][i][pos[0]][pos[1]] = _check_condition(
               piece, role, team, hidden, flip_teams);
         }
      }
   }
   return board_state_rep;
}

template < typename Board >
std::vector< unsigned int > RepresenterStratego::get_action_mask_(const Board &board, int player)
{
   return get_action_mask_(m_actions, board, player);
}

template < typename Piece >
bool RepresenterStratego::_check_condition(
   const sptr< Piece > &piece,
   const role_type &role,
   int team,
   bool hidden,
   bool flip_teams) const
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
         bool eq_role = piece->get_role() == role;
         return eq_team && eq_role;
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
            bool eq_role = piece->get_role() == role;
            return eq_team && eq_role;
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
std::vector< unsigned int > RepresenterStratego::get_action_mask_(
   const std::vector< action_type > &actions, const Board &board, int player)
{
   std::vector< unsigned int > action_mask(actions.size(), 0);
   for(const auto &action : actions) {
      position_type old_pos{0, 0};
      if(auto pos_pointer = board.get_position_of_role(player, action.get_assoc_role());
         pos_pointer == board.end_inverse(player)) {
         continue;
      } else
         old_pos = pos_pointer->second;
      position_type new_pos = old_pos + action.get_effect();
      if(LogicStratego< board_type >::is_legal_move_(board, {old_pos, new_pos})) {
         action_mask[action.get_index()] = 1;
      }
   }
   return action_mask;
}
