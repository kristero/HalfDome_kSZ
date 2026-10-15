#include <cstdio>
#include <cstddef>
#include <array>
struct S {
    static constexpr const size_t Ncells_side = 64UL;
    std::array<int,3> f(int xx, int yy, int zz) const {
        #define PER(x) ((x>=Ncells_side) ? 1 : (x<0) ? -1 : 0)
        std::array<int,3> p { PER(xx), PER(yy), PER(zz) };
        #undef PER
        return p;
    }
};
int main(){
    S s;
    for (int xx : {-3,-2,-1,0,1,62,63,64,65}) {
        auto p = s.f(xx, 0, 0);
        std::printf("xx=%3d -> PER=%+d\n", xx, p[0]);
    }
    // the distance with the hint, for a group at x_g = 0.3 acell and particle in cell 63 (x_p = 63.5 acell), L = 64 acell
    double acell = 205000.0/64, L = 205000.0;
    double xg = 0.3*acell, xp = 63.5*acell;
    int hint = s.f(-1,0,0)[0];
    std::printf("hint=%d dx_hint=%.1f ckpc/h  (true min-image dx=%.1f)\n", hint, xp - xg + L*hint, xp - xg - L);
    return 0;
}
