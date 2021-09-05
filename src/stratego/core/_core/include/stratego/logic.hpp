
#ifndef STRATEGO_LOGIC_H
#define STRATEGO_LOGIC_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <vector>

namespace py = pybind11;

/**
 * Abstract Base Class for the game logic
 */
class Logic {

    public:
    Logic() = delete;
    virtual ~Logic() = default;

    virtual std::optional<int> execute_move(State& state, Move& move);
};


#endif //STRATEGO_LOGIC_H
