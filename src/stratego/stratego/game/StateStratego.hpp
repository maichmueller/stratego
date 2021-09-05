
#pragma once

#include <unordered_set>

#include "aze/game.h"
#include "board/BoardStratego.h"
#include "logic/LogicStratego.h"

class StateStratego: public State< BoardStratego > {
  public:
   using base_type = State< BoardStratego >;

   // just decorate all base constructors with initializing also the dead pieces
   // variable.
   template < typename... Params >
   StateStratego(Params &&... params)
       : base_type(std::forward< Params >(params)...), m_dead_pieces()
   {
   }

   // also declare some explicit constructors
   explicit StateStratego(size_t shape_x, size_t shape_y);

   explicit StateStratego(size_t shape = 5);

   StateStratego(
      size_t shape,
      const std::map< position_type, role_type > &setup_0,
      const std::map< position_type, role_type > &setup_1);

   StateStratego(
      std::array< size_t, 2 > shape,
      const std::map< position_type, role_type > &setup_0,
      const std::map< position_type, role_type > &setup_1);

   StateStratego(
      size_t shape,
      const std::map< position_type, int > &setup_0,
      const std::map< position_type, int > &setup_1);

   StateStratego(
      std::array< size_t, 2 > shape,
      const std::map< position_type, int > &setup_0,
      const std::map< position_type, int > &setup_1);

   int _do_move(const move_type &move) override;

  protected:
   static int fight(piece_type &attacker, piece_type &defender);

  private:
   using dead_pieces_type = std::array< std::unordered_set< role_type >, 2 >;
   dead_pieces_type m_dead_pieces;

   void _update_dead_pieces(const sptr< piece_type > &piece)
   {
      if(! piece->is_null())
         m_dead_pieces[piece->get_team()].emplace(piece->get_role());
   }

   StateStratego *clone_impl() const override;
};
