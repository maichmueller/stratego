//
// Created by Michael on 08.09.19.
//

#include "BoardStratego.h"

#include "logic/LogicStratego.h"

#define VERT_BAR "\u2588"
#define RESET "\x1B[0m"
#define BLUE "\x1B[44m"
#define RED "\x1B[41m"

std::vector< sptr< typename BoardStratego::piece_type > > BoardStratego::adapt_setup(
   const std::map< position_type, int > &setup)
{
   std::vector< sptr< BoardStratego::piece_type > > vector_out;

   std::map< position_type, int > seen_pos;
   std::map< int, int > version_count;
   for(auto &elem : setup) {
      position_type pos = elem.first;
      auto type = elem.second;

      if(seen_pos.find(pos) != seen_pos.end()) {
         // element found
         throw std::invalid_argument(
            "Parameter setup has more than one piece for the "
            "same position (position: '"
            + pos.to_string() + "').");
      }
      seen_pos[pos] = 1;
      // null constructor of map is called on unplaced previous item (creates 0
      // int) therefore the first time this is called, will get us to version 0,
      // the first piece of its kind. Afterwards it will keep the count
      // correctly for us.
      int version = version_count[type]++;
      vector_out.push_back(std::make_shared< piece_type >(pos, role_type(type, version), 0));
   }
   return vector_out;
}

std::string BoardStratego::print_board(int player, bool hide_unknowns) const
{
   int H_SIZE_PER_PIECE = 9;
   int V_SIZE_PER_PIECE = 3;
   // the space needed to assign row indices to the rows and to add a splitting
   // bar "|"
   int row_ind_space = 4;
   int dim_x = m_shape[0];
   int dim_y = m_shape[1];
   int mid = V_SIZE_PER_PIECE / 2;

   // piece string lambda function that returns a str of the kin
   // "-1 \n
   // 10.1 \n
   //   1"
   auto create_piece_str = [&H_SIZE_PER_PIECE, &mid, &player, &hide_unknowns](
                              const piece_type &piece, int line) {
      if(piece.is_null())
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');

      std::string color = BLUE;  // blue by default (for player 0)
      if(piece.get_team() == -1 && ! piece.is_null())
         // piece is an obstacle (return a gray colored field)
         return "\x1B[30;47m" + utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
      else if(piece.get_team(player) == 1) {
         color = RED;  // background red, text "white"
      }
      if(line == mid - 1) {
         // hidden info line
         std::string h = piece.get_flag_hidden() ? "?" : " ";
         return color + utils::center(h, H_SIZE_PER_PIECE, " ") + RESET;
      } else if(line == mid) {
         // type and version info line
         if(hide_unknowns && piece.get_flag_hidden() && piece.get_team(player)) {
            return color + std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ') + RESET;
         }
         const auto &role = piece.get_role();
         return color
                + utils::center(
                   std::to_string(role[0]) + '.' + std::to_string(role[1]), H_SIZE_PER_PIECE, " ")
                + RESET;
      } else if(line == mid + 1)
         // team info line
         // return color + center(std::to_string(piece.get_team(flip_board)),
         // H_SIZE_PER_PIECE, " ") + reset;
         return color + utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
      else
         // empty line
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
   };

   std::stringstream board_print;
   board_print << "\n";

   std::string init_space = std::string(static_cast< unsigned long >(row_ind_space), ' ');
   std::string h_border = utils::repeat(
      VERT_BAR, static_cast< unsigned long >(dim_x * (H_SIZE_PER_PIECE + 1) - 1));

   board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   std::string init = board_print.str();
   sptr< piece_type > curr_piece;

   // row means row of the board. not actual rows of console output.
   for(int row = dim_y - 1; row > m_starts[1] - 1;
       --row) {  // iterate backwards through the rows for correct display
      // per piece we have V_SIZE_PER_PIECE many lines to fill consecutively.
      // Iterate over every column and append the new segment to the right line.
      std::vector< std::stringstream > line_streams(static_cast< unsigned int >(V_SIZE_PER_PIECE));

      for(int col = m_starts[0]; col < dim_x; ++col) {
         if(player) {
            curr_piece = (*this)[{dim_x - 1 - row, dim_y - 1 - col}];
         } else
            curr_piece = (*this)[{row, col}];

         for(int i = 0; i < V_SIZE_PER_PIECE; ++i) {
            std::stringstream curr_stream;

            if(i == mid - 1 || i == mid + 1) {
               if(col == 0) {
                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space), ' ');
               }
               curr_stream << VERT_BAR << create_piece_str(*curr_piece, i);
            } else if(i == mid) {
               if(col == 0) {
                  if(row < 10)
                     curr_stream << " " << row;
                  else
                     curr_stream << row;

                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space - 2), ' ')
                              << VERT_BAR;
               }
               curr_stream << create_piece_str(*curr_piece, i);
               if(col != dim_x - 1)
                  curr_stream << VERT_BAR;
            }
            // extend the current line i by the new information
            line_streams[i] << curr_stream.str();
         }
      }
      for(auto &stream : line_streams) {
         board_print << stream.str() << VERT_BAR << "\n";
      }
      board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   }
   // column width for the row index plus vertical dash
   board_print << std::string(static_cast< unsigned long >(row_ind_space), ' ');
   // print the column index rows
   for(int i = m_starts[0]; i < dim_x; ++i) {
      board_print << utils::center(std::to_string(i), H_SIZE_PER_PIECE + 1, " ");
   }
   board_print << "\n";
   return board_print.str();
}

void BoardStratego::_add_obstacles()
{
   auto obstacle_positions = LogicStratego< BoardStratego >::get_obstacle_positions(m_shape[0]);
   for(const auto &obstacle_pos : obstacle_positions) {
      m_map[obstacle_pos] = std::make_shared< piece_type >(obstacle_pos, role_type{99, 99}, -1);
   }
}

BoardStratego *BoardStratego::clone_impl() const
{
   auto *board_copy_ptr = new BoardStratego(*this);
   for(auto &sptr : *board_copy_ptr) {
      sptr.second = std::make_shared< piece_type >(*sptr.second);
   }
   return board_copy_ptr;
}