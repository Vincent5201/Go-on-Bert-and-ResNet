#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <set>
#include <queue>
#include <algorithm>

namespace py = pybind11;

const int BOARD_SIZE = 19;

inline bool valid_pos(int x, int y) {
    return x >= 0 && y >= 0 && x < BOARD_SIZE && y < BOARD_SIZE;
}

bool value_board(py::array_t<int> board_in) {
    auto buf = board_in.request();
    if (buf.ndim != 3 || buf.shape[0] != 4 || buf.shape[1] != BOARD_SIZE || buf.shape[2] != BOARD_SIZE)
        throw std::runtime_error("Invalid board shape");

    int (*board2)[BOARD_SIZE][BOARD_SIZE] = new int[4][BOARD_SIZE][BOARD_SIZE];
    auto ptr = static_cast<int*>(buf.ptr);
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < BOARD_SIZE; ++i)
            for (int j = 0; j < BOARD_SIZE; ++j)
                board2[k][i][j] = ptr[k * BOARD_SIZE * BOARD_SIZE + i * BOARD_SIZE + j];

    auto neighbor_liberty = [&](int p, int x, int y) {
        int pp = 1 - p;
        std::set<std::pair<int, int>> counted;
        std::vector<std::pair<int, int>> stack = {{x, y}};
        int liberty = BOARD_SIZE * BOARD_SIZE;

        while (!stack.empty()) {
            auto [cx, cy] = stack.back(); stack.pop_back();
            counted.insert({cx, cy});
            for (auto [dx, dy] : std::vector<std::pair<int, int>>{{cx+1,cy}, {cx-1,cy}, {cx,cy+1}, {cx,cy-1}}) {
                if (valid_pos(dx, dy) && !counted.count({dx, dy})) {
                    if (board2[p][dx][dy])
                        stack.push_back({dx, dy});
                    else if (board2[pp][dx][dy])
                        liberty = std::min(liberty, board2[3][dx][dy]);
                }
            }
        }
        return liberty;
    };

    std::function<void(int, int, int)> del_die = [&](int x, int y, int p) {
        board2[p][x][y] = 0;
        board2[3][x][y] = 0;
        for (auto [dx, dy] : std::vector<std::pair<int, int>>{{x-1,y}, {x,y-1}, {x+1,y}, {x,y+1}}) {
            if (valid_pos(dx, dy) && board2[p][dx][dy])
                del_die(dx, dy, p);
        }
    };

    std::function<std::pair<int, int>(int, int)> count_neighbor = [&](int x, int y) {
        std::set<std::pair<int, int>> counted;
        std::function<std::pair<int, int>(int, int, int)> dfs = [&](int x, int y, int dist) -> std::pair<int, int> {
            if (dist == 0) return {0, 0};
            counted.insert({x, y});
            int p0 = 0, p1 = 0;
            for (auto [dx, dy] : std::vector<std::pair<int, int>>{{x+1,y}, {x,y-1}, {x-1,y}, {x,y+1}}) {
                if (valid_pos(dx, dy) && !counted.count({dx, dy})) {
                    if (board2[0][dx][dy]) p0++;
                    else if (board2[1][dx][dy]) p1++;
                    else {
                        auto [t0, t1] = dfs(dx, dy, dist - 1);
                        p0 += t0;
                        p1 += t1;
                    }
                }
            }
            return {p0, p1};
        };
        return dfs(x, y, 10);
    };

    for (int i = 0; i < BOARD_SIZE; ++i)
        for (int j = 0; j < BOARD_SIZE; ++j)
            if (board2[3][i][j] == 1) {
                int p = board2[1][i][j] ? 1 : 0;
                if (neighbor_liberty(p, i, j) > 1)
                    del_die(i, j, p);
            }

    int p0 = 0, p1 = 0;
    for (int i = 0; i < BOARD_SIZE; ++i)
        for (int j = 0; j < BOARD_SIZE; ++j)
            if (board2[0][i][j] == 0 && board2[1][i][j] == 0) {
                auto [t0, t1] = count_neighbor(i, j);
                p0 += t0;
                p1 += t1;
            }

    delete[] board2;
    return p1 > p0 + 5;
}

void channel_01(py::array_t<int> board, int x, int y, int turn) {
    auto b = board.mutable_unchecked<3>();

    std::set<std::pair<int, int>> live, died;

    std::function<bool(int, int, int)> checkDie = [&](int x, int y, int p) -> bool {
        int pp = 1 - p;
        if (live.count({x, y})) return false;
        if (died.count({x, y})) return true;
        died.insert({x, y});
        bool ans = true;
        std::vector<std::pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy)) {
                if (b(p, dx, dy) == 0 && b(pp, dx, dy) == 0) {
                    live.insert({x, y});
                    return false;
                }
                if (b(p, dx, dy) == 1) {
                    ans = ans & checkDie(dx, dy, p);
                }
            }
        }
        if (!ans) {
            died.erase({x, y});
            live.insert({x, y});
        }
        return ans;
    };

    std::function<void(int, int, int)> del_die = [&](int x, int y, int p) {
        b(p, x, y) = 0;
        b(3, x, y) = 0;
        std::vector<std::pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy) && b(p, dx, dy)) {
                del_die(dx, dy, p);
            }
        }
    };

    b(turn % 2, x, y) = 1;

    std::vector<std::pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
    for (auto [dx, dy] : directions) {
        if (valid_pos(dx, dy)) {
            if (turn % 2 == 1 && b(0, dx, dy) && checkDie(dx, dy, 0)) {
                del_die(dx, dy, 0);
            } else if (turn % 2 == 0 && b(1, dx, dy) && checkDie(dx, dy, 1)) {
                del_die(dx, dy, 1);
            }
        }
    }
}

void channel_3(py::array_t<int> board, int x, int y, int turn) {
    // Access the mutable board (3D numpy array)
    auto b = board.mutable_unchecked<3>();

    std::set<std::pair<int, int>> counted_empty;
    std::set<std::pair<int, int>> counted_pos;

    // Recursive function to check liberties
    std::function<int(int, int, int)> check_liberty = [&](int x, int y, int p) -> int {
        int liberty = 0;
        int pp = (p == 0) ? 1 : 0;
        b(p, x, y) = 2;  // Temporarily mark the position to avoid infinite loops
        std::vector<std::pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy)) {
                if (b(pp, dx, dy) == 0 && b(p, dx, dy) == 0) {
                    if (counted_empty.find({dx, dy}) == counted_empty.end()) {
                        liberty += 1;
                        counted_empty.insert({dx, dy});
                    }
                } else if (b(p, dx, dy) == 1) {
                    liberty += check_liberty(dx, dy, p);
                }
            }
        }

        b(p, x, y) = 1;  // Restore the position
        counted_pos.insert({x, y});
        return liberty;
    };

    // Function to set liberties
    std::function<void(int, int, int, int)> set_liberty = [&](int x, int y, int p, int liberty) {
        b(p, x, y) = 2;
        b(3, x, y) = std::min(6, liberty);  // Store liberty value in the third plane
        std::vector<std::pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
        
        for (auto [dx, dy] : directions) {
            if (valid_pos(dx, dy) && b(p, dx, dy) == 1) {
                set_liberty(dx, dy, p, liberty);
            }
        }
        b(p, x, y) = 1;  // Restore the position
    };

    // If both positions are empty, return early
    if (b(0, x, y) == 0 && b(1, x, y) == 0) {
        return;
    }

    // Set liberties for the current position
    set_liberty(x, y, turn % 2, check_liberty(x, y, turn % 2));

    int pp = (turn % 2 == 0) ? 1 : 0;
    std::vector<std::pair<int, int>> directions = {{x-1, y}, {x, y-1}, {x+1, y}, {x, y+1}};
    
    // Iterate over adjacent positions
    for (auto [dx, dy] : directions) {
        counted_empty.clear();
        if (valid_pos(dx, dy) && b(pp, dx, dy) && counted_pos.find({dx, dy}) == counted_pos.end()) {
            set_liberty(dx, dy, pp, check_liberty(dx, dy, pp));
        }
    }
}

PYBIND11_MODULE(cpptools, m) {
    m.def("value_board", &value_board, "value board");
    m.def("channel_01", &channel_01, "channel_01");
    m.def("channel_3", &channel_3, "channel_3");
}
