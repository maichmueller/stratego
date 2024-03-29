
#pragma once

#include <functional>

#include "aze/board.h"
#include "aze/logic.h"
#include "board/BoardStratego.h"

struct BattleMatrix {
   static const std::map< std::array< int, 2 >, int > battle_matrix;

   static std::map< std::array< int, 2 >, int > initialize_battle_matrix();

   [[nodiscard]] static int fight_outcome(std::array< int, 2 > att_def)
   {
      return battle_matrix.at(att_def);
   }
};

static constexpr const size_t MAX_REPEATED_ROUNDS = 3;

template < class BoardType >
struct LogicStratego: public Logic< BoardType, LogicStratego< BoardType > > {
   using base_type = Logic< BoardType, LogicStratego< BoardType > >;
   using board_type = typename base_type::board_type;
   using move_type = typename base_type::move_type;
   using position_type = typename base_type::position_type;
   using piece_type = typename board_type::piece_type;
   using role_type = typename board_type::role_type;

   static bool is_legal_move_(const board_type &board, const move_type &move);

   static std::vector< move_type > get_legal_moves_(
      const board_type &board, int player, bool flip_board = false);

   static bool has_legal_moves_(const board_type &board, int player);

   static auto get_battle_matrix() { return BattleMatrix::battle_matrix; }

   static int fight_outcome(piece_type attacker, piece_type defender)
   {
      return fight_outcome(std::array{attacker.get_role()[0], defender.get_role()[0]});
   }

   static int fight_outcome(std::array< int, 2 > att_def)
   {
      return BattleMatrix::fight_outcome(att_def);
   }

   static inline std::vector< position_type > get_obstacle_positions(int shape)
   {
      if(shape == 5)
         return {{2, 2}};
      else if(shape == 7)
         return {{3, 1}, {3, 5}};
      else if(shape == 10)
         return {{4, 2}, {5, 2}, {4, 3}, {5, 3}, {4, 6}, {5, 6}, {4, 7}, {5, 7}};
      else
         throw std::invalid_argument("'shape' not in {5, 7, 10}.");
   }

   static inline std::vector< int > get_available_types(int shape)
   {
      if(shape == 5)
         return {0, 1, 2, 2, 2, 3, 3, 10, 11, 11};
      else if(shape == 7)
         return {0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 10, 11, 11, 11, 11};
      else if(shape == 10)
         return {0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,  3,  4,  4,  4,  4,  5,
                 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10, 11, 11, 11, 11, 11, 11};
      else
         throw std::invalid_argument("'shape' not in {5, 7, 10}.");
   }

   static inline std::vector< position_type > get_start_positions(int shape, int team)
   {
      if(team != 0 && team != 1)
         throw std::invalid_argument("'team' not in {0, 1}.");

      if(shape == 5) {
         if(team == 0)
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}};
         else
            return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}};
      } else if(shape == 7) {
         if(team == 0)
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6},
                    {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6},
                    {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}};
         else
            return {{4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
                    {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6},
                    {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}};

      } else if(shape == 10) {
         if(team == 0)
            return {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 9},
                    {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9},
                    {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8}, {2, 9},
                    {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {3, 8}, {3, 9}};

         else
            return {{6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}, {6, 7}, {6, 8}, {6, 9},
                    {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}, {7, 7}, {7, 8}, {7, 9},
                    {8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5}, {8, 6}, {8, 7}, {8, 8}, {8, 9},
                    {9, 0}, {9, 1}, {9, 2}, {9, 3}, {9, 4}, {9, 5}, {9, 6}, {9, 7}, {9, 8}, {9, 9}};

      } else
         throw std::invalid_argument("'shape' not in {5, 7, 10}.");
   }
};

template < typename BoardType >
bool LogicStratego< BoardType >::is_legal_move_(const board_type &board, const move_type &move)
{
   const auto &[pos_before, pos_after] = move.get_positions();

   if(! (std::get< 0 >(board.check_bounds(pos_before)))
      || ! (std::get< 0 >(board.check_bounds(pos_after))))
      return false;

   sptr< piece_type > p_b = board[pos_before];
   sptr< piece_type > p_a = board[pos_after];

   if(p_b->is_null())
      return false;
   if(int type = p_b->get_role()[0]; type == 0 || type == 11) {
      return false;
   }
   if(! p_a->is_null()) {
      if(p_a->get_team() == p_b->get_team())
         return false;  // cant fight pieces of own team
      if(p_a->get_role()[0] == 99) {
         return false;  // cant fight obstacle
      }
   }

   int move_dist = abs(pos_after[1] - pos_before[1]) + abs(pos_after[0] - pos_before[0]);
   if(move_dist > 1) {
      if(p_b->get_role()[0] != 2)
         return false;  // not of type 2 , but is supposed to go far

      if(pos_after[0] == pos_before[0]) {
         int dist = pos_after[1] - pos_before[1];
         int sign = (dist >= 0) ? 1 : -1;
         for(int i = 1; i < std::abs(dist); ++i) {
            position_type pos{pos_before[0], pos_before[1] + sign * i};
            if(! board[pos]->is_null())
               return false;
         }
      } else if(pos_after[1] == pos_before[1]) {
         int dist = pos_after[0] - pos_before[0];
         int sign = (dist >= 0) ? 1 : -1;
         for(int i = 1; i < std::abs(dist); ++i) {
            position_type pos = {pos_before[0] + sign * i, pos_before[1]};
            if(! board[pos]->is_null())
               return false;
         }
      } else
         return false;  // diagonal moves not allowed
   }
   return true;
}

template < class BoardType >
std::vector< typename LogicStratego< BoardType >::move_type >
LogicStratego< BoardType >::get_legal_moves_(const board_type &board, int player, bool flip_board)
{
   int shape_x = board.get_shape()[0];
   int shape_y = board.get_shape()[1];
   int starts_x = board.get_starts()[0];
   int starts_y = board.get_starts()[1];
   std::vector< move_type > moves_possible;
   for(auto elem = board.begin(); elem != board.end(); ++elem) {
      sptr< piece_type > piece = elem->second;
      if(! piece->is_null() && piece->get_team() == player) {
         // the position we are dealing with
         Position pos = piece->get_position();

         if(piece->get_role()[0] == 2) {
            // all possible moves to the right until board ends
            for(int i = 1; i < starts_x + shape_x - pos[0]; ++i) {
               position_type pos_to{pos[0] + i, pos[1]};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  moves_possible.push_back(move);
               }
            }
            // all possible moves to the top until board ends
            for(int i = 1; i < starts_y + shape_y - pos[1]; ++i) {
               position_type pos_to{pos[0], pos[1] + i};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  moves_possible.push_back(move);
               }
            }
            // all possible moves to the left until board ends
            for(int i = 1; i < starts_x + pos[0] + 1; ++i) {
               position_type pos_to{pos[0] - i, pos[1]};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  moves_possible.push_back(move);
               }
            }
            // all possible moves to the bottom until board ends
            for(int i = 1; i < starts_y + pos[1] + 1; ++i) {
               position_type pos_to{pos[0], pos[1] - i};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  moves_possible.push_back(move);
               }
            }
         } else {
            // all moves are 1 step to left, right, top, or bottom
            std::vector< position_type > pos_tos = {
               {pos[0] + 1, pos[1]},
               {pos[0], pos[1] + 1},
               {pos[0] - 1, pos[1]},
               {pos[0], pos[1] - 1}};
            for(auto &pos_to : pos_tos) {
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  moves_possible.push_back(move);
               }
            }
         }
      }
   }
   if(flip_board) {
      auto shape = board.get_shape();
      auto starts = board.get_starts();
      for(auto &move : moves_possible) {
         move = move.invert(starts, shape);
      }
   }
   return moves_possible;
}

template < class BoardType >
bool LogicStratego< BoardType >::has_legal_moves_(const board_type &board, int player)
{
   int shape_x = board.get_shape()[0];
   int shape_y = board.get_shape()[1];
   int starts_x = board.get_starts()[0];
   int starts_y = board.get_starts()[1];
   for(auto elem = board.begin(); elem != board.end(); ++elem) {
      sptr< piece_type > piece = elem->second;
      if(int essential_role = piece->get_role()[0];
         ! piece->is_null() && piece->get_team() == player && essential_role != 0
         && essential_role != 11) {
         // the position we are dealing with
         Position pos = piece->get_position();

         if(piece->get_role()[0] == 2) {
            // all possible moves to the right until board ends
            for(int i = 1; i < starts_x + shape_x - pos[0]; ++i) {
               position_type pos_to{pos[0] + i, pos[1]};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  return true;
               }
            }
            // all possible moves to the top until board ends
            for(int i = 1; i < starts_y + shape_y - pos[1]; ++i) {
               position_type pos_to{pos[0], pos[1] + i};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  return true;
               }
            }
            // all possible moves to the left until board ends
            for(int i = 1; i < starts_x + pos[0] + 1; ++i) {
               position_type pos_to{pos[0] - i, pos[1]};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  return true;
               }
            }
            // all possible moves to the bottom until board ends
            for(int i = 1; i < starts_y + pos[1] + 1; ++i) {
               position_type pos_to{pos[0], pos[1] - i};
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  return true;
               }
            }
         } else {
            // all moves are 1 step to left, right, top, or bottom
            std::vector< position_type > pos_tos = {
               {pos[0] + 1, pos[1]},
               {pos[0], pos[1] + 1},
               {pos[0] - 1, pos[1]},
               {pos[0], pos[1] - 1}};
            for(const auto &pos_to : pos_tos) {
               move_type move{pos, pos_to};
               if(is_legal_move_(board, move)) {
                  return true;
               }
            }
         }
      }
   }
   return false;
}
