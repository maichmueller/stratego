#pragma once

#include "array"
#include "aze/board.h"

class PieceStratego: public Piece< Position< int, 2 >, 2 > {
  public:
   using base_type = Piece< Position< int, 2 >, 2 >;

   // inheriting the base constructors with the next command!
   using base_type::base_type;
};