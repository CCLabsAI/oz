#include <catch.hpp>

#include "game.h"
#include "games/holdem.h"

#include "ace_eval.h"

using oz::holdem_poker_t;
using oz::CHANCE;
using oz::P1;
using oz::P2;

using std::begin;
using std::end;

using card_t = holdem_poker_t::card_t;
using action_t = holdem_poker_t::action_t;
using hand_t = holdem_poker_t::hand_t;
using board_t = holdem_poker_t::board_t;
using phase_t = holdem_poker_t::phase_t;

TEST_CASE("holdem poker deal utilities", "[holdem]") {
  for (card_t c1 = holdem_poker_t::CARD_MIN; c1 < holdem_poker_t::N_CARDS; c1++) {
    holdem_poker_t::action_t a = holdem_poker_t::deal_action_for_card(c1);
    card_t c2 = holdem_poker_t::card_for_deal_action(a);
    CHECK(holdem_poker_t::is_deal_action(a));
    CHECK(c1 == c2);
  }

  CHECK(!holdem_poker_t::is_deal_action(action_t::Raise));
  CHECK(!holdem_poker_t::is_deal_action(action_t::Call));
  CHECK(!holdem_poker_t::is_deal_action(action_t::Fold));
}

static inline void deal_hand(holdem_poker_t& game, const oz::player_t player, const hand_t& hand) {
  CHECK(game.player() == CHANCE);

  for (const auto c : hand) {
    auto a = make_action(holdem_poker_t::deal_action_for_card(c));
    auto action_probs = game.chance_actions();
    CHECK(action_probs.find(a)->second > 0);
    game.act(a);
  }

  CHECK(game.hand(player) == hand);
}

static inline void deal_board(holdem_poker_t& game, const board_t& board) {
  CHECK(game.player() == CHANCE);
  board_t b = game.board();

  for (const auto c : board) {
    auto a = make_action(holdem_poker_t::deal_action_for_card(c));
    auto action_probs = game.chance_actions();
    CHECK(action_probs.find(a)->second > 0);
    game.act(a);
  }

  b.insert(end(b), begin(board), end(board));
  CHECK(game.board() == b);
}

TEST_CASE("holdem poker basic actions", "[holdem]") {
  using namespace oz::poker_cards;

  auto game = holdem_poker_t();

  CHECK(game.phase() == holdem_poker_t::phase_t::DEAL_HOLE_P1);
  deal_hand(game, P1, {{_Td, _As}});
  CHECK(game.chance_actions().size() == 50);

  CHECK(game.phase() == holdem_poker_t::phase_t::DEAL_HOLE_P2);
  deal_hand(game, P2, {{_3c, _8d}});
  CHECK(game.phase() == holdem_poker_t::phase_t::BET);

  CHECK(game.player() == P2);
  CHECK(game.pot(P1) == 10);
  CHECK(game.pot(P2) == 5);
  game.act(make_action(action_t::Call));

  CHECK(game.player() == P1);
  game.act(make_action(action_t::Raise));
  CHECK(game.pot(P1) == 20);

  CHECK(game.player() == P2);
  game.act(make_action(action_t::Raise));
  CHECK(game.pot(P2) == 30);

  CHECK(game.player() == P1);
  game.act(make_action(action_t::Call));
  CHECK(game.phase() == holdem_poker_t::phase_t::DEAL_BOARD);
  CHECK(game.player() == CHANCE);
  CHECK(game.round() == 1);

  CHECK(game.board().empty());

  deal_board(game, {_6h, _3h, _9s});

  CHECK(game.board().size() == 3);
  game.act(make_action(action_t::Raise));
  game.act(make_action(action_t::Raise));
  game.act(make_action(action_t::Raise));
  game.act(make_action(action_t::Raise));

  CHECK(game.infoset().actions().size() == 2); // can't raise

  game.act(make_action(action_t::Call));
  CHECK(game.phase() == holdem_poker_t::phase_t::DEAL_BOARD);
  CHECK(game.player() == CHANCE);

  CHECK(game.str() == "TdAs|3c8d/6h3h9s:crrc/rrrrc/");

  deal_board(game, {_Ah});
  CHECK(game.phase() == holdem_poker_t::phase_t::BET);
  CHECK(game.player() == P1);

  CHECK(game.infoset().str() == "TdAs/6h3h9sAh:crrc/rrrrc/");

  game.act(make_action(action_t::Call));
  game.act(make_action(action_t::Call));

  CHECK(game.phase() == holdem_poker_t::phase_t::DEAL_BOARD);
  deal_board(game, {_5c});

  game.act(make_action(action_t::Raise));
  game.act(make_action(action_t::Call));

  CHECK(game.is_terminal());
  CHECK(game.phase() == holdem_poker_t::phase_t::FINISHED);

  CHECK(game.utility(P1) == 90);
}

static inline void check_hand(int a, int b, int c, int d, int e, int f, int g, ace::Card rank)
{
  using namespace ace;

  Card ar;
  Card h[5] = { };

  ACE_addcard(h,ACE_makecard(a));
  ACE_addcard(h,ACE_makecard(b));
  ACE_addcard(h,ACE_makecard(c));
  ACE_addcard(h,ACE_makecard(d));
  ACE_addcard(h,ACE_makecard(e));
  ACE_addcard(h,ACE_makecard(f));
  ACE_addcard(h,ACE_makecard(g));
  CHECK(ACE_evaluate(h) == rank);

  CHECK(holdem_poker_t::hand_rank({{a,b}}, {c,d,e,f,g}) == rank);
}

TEST_CASE("holdem poker hand rank", "[holdem]") {
  using namespace ace;
  using namespace oz::poker_cards;

  CHECK(ACE_makecard(_2h) == 0x00000041);
  CHECK(ACE_makecard(_3h) == 0x00000101);
  CHECK(ACE_makecard(_Kh) == 0x10000001);
  CHECK(ACE_makecard(_Ah) == 0x40000001);
  CHECK(ACE_makecard(_2c) == 0x00000042);
  CHECK(ACE_makecard(_3c) == 0x00000102);
  CHECK(ACE_makecard(_Kc) == 0x10000002);
  CHECK(ACE_makecard(_Ac) == 0x40000002);
  CHECK(ACE_makecard(_2d) == 0x00000044);
  CHECK(ACE_makecard(_3d) == 0x00000104);
  CHECK(ACE_makecard(_Kd) == 0x10000004);
  CHECK(ACE_makecard(_Ad) == 0x40000004);
  CHECK(ACE_makecard(_2s) == 0x00000048);
  CHECK(ACE_makecard(_3s) == 0x00000108);
  CHECK(ACE_makecard(_Ks) == 0x10000008);
  CHECK(ACE_makecard(_As) == 0x40000008);

  //a kqjt 9876 5432
  //no hand
  check_hand(_2h,_3h,_4h,_5h,_Td,_Jd,_Kd, 0<<28|0x0000<<13|0x0B0C);
  //1pr
  check_hand(_Kh,_3h,_4h,_5h,_Ad,_Jd,_Kd, 1<<28|0x0800<<13|0x1208);
  //2pr
  check_hand(_Kd,_3h,_3c,_5h,_Ad,_Jd,_Ks, 2<<28|0x0802<<13|0x1000);
  check_hand(_Kd,_3h,_3c,_5h,_5s,_2d,_Ks, 2<<28|0x0808<<13|0x0002);
  //trip
  check_hand(_3d,_4d,_6h,_3h,_9s,_3c,_Td, 3<<28|0x0002<<13|0x0180);
  //straight
  check_hand(_3d,_4d,_9h,_5h,_2d,_3c,_Ad, 4<<28|0x0008<<13|0x0000);
  check_hand(_3d,_4d,_6h,_5h,_2d,_3c,_Ad, 4<<28|0x0010<<13|0x0000);
  check_hand(_3d,_4d,_6h,_5h,_2d,_7c,_Ad, 4<<28|0x0020<<13|0x0000);
  check_hand(_Td,_Jd,_Kh,_5h,_Qd,_3c,_Ad, 4<<28|0x1000<<13|0x0000);
  //flush
  check_hand(_2h,_3h,_4h,_5h,_Td,_Jh,_Kd, 5<<28|0x020F<<13|0x0000);
  check_hand(_Kh,_3h,_4h,_5h,_Ah,_Jd,_Kd, 5<<28|0x180E<<13|0x0000);
  check_hand(_Kd,_3h,_3d,_5d,_Ad,_Jd,_Ks, 5<<28|0x1A0A<<13|0x0000);
  check_hand(_3d,_4d,_6d,_3h,_9d,_3c,_Td, 5<<28|0x0196<<13|0x0000);
  check_hand(_3d,_4d,_6d,_3h,_9d,_2d,_Td, 5<<28|0x0196<<13|0x0000);
  check_hand(_3d,_4d,_6d,_7d,_9d,_Jd,_Td, 5<<28|0x03B0<<13|0x0000);
  //fh
  check_hand(_3d,_4d,_3h,_4h,_9s,_3c,_Td, 6<<28|0x0002<<13|0x0004);
  check_hand(_4s,_4d,_6h,_4h,_3s,_3c,_Td, 6<<28|0x0004<<13|0x0002);
  check_hand(_3d,_9d,_9h,_3h,_9s,_3c,_Td, 6<<28|0x0080<<13|0x0002);

  //quad
  check_hand(_3d,_9d,_9h,_3h,_3s,_3c,_Td, 7<<28|0x0002<<13|0x0100);

  //straight flush low
  check_hand(_Ah,_2h,_3h,_4h,_5h,_9d,_Td, 9<<28|0x0008<<13|0x0000);
  check_hand(_Ad,_2h,_3h,_Jd,_Kd,_Qd,_Td, 9<<28|0x1000<<13|0x0000);
  check_hand(_9d,_7d,_8d,_Jd,_Kd,_Qd,_Td, 9<<28|0x0800<<13|0x0000);
}

static holdem_poker_t check_read_print(std::string s) {
  holdem_poker_t game;
  game.read_history_str(s);
  CHECK(game.str() == s);
  return game;
}

TEST_CASE("holdem poker history reader", "[holdem]") {
  using namespace oz::poker_cards;

  check_read_print("TdAs|????");
  check_read_print("TdAs|3c8d");
  check_read_print("TdAs|3c8d:crr");
  check_read_print("TdAs|3c8d/6h3h9s:crrc/");

  auto game = check_read_print("TdAs|3c8d/6h3h9s2d:crrc/rrrrc/");
  CHECK(game.hand(P1) == hand_t {{ _Td, _As }});
  CHECK(game.hand(P2) == hand_t {{ _3c, _8d }});
  CHECK(game.phase() == phase_t::BET);
  CHECK(game.board() == board_t { _6h, _3h, _9s, _2d });

  check_read_print("TdAs|3c8d/6h3h9sTs2d:crrc/crrrrc/crrrrc/crrrrc/");
}
