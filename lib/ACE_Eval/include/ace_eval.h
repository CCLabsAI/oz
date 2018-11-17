namespace ace {

#include <cstdint>

typedef uint32_t Card;

static constexpr int ACEHAND = 5;

static inline Card ACE_makecard(int i) {
  return 1<<(2*(i%13)+6)|1<<(i/13);
}

/* Add a card to the hand:
   The hand is stored as an array of 5 ints. The low 3 bits are used to select a suit,
   (h[0]=spade,h[1]=club,h[2]=diamond,h[4]=heart).
  The cards are added to the suit - since only one suit bit is set, 3 of the low 8 bits
   will contain a count of the cards in that suit (that's why the spare is where it is).
     h[3] has a single bit set for each card present, used for straight detection
*/
static inline void ACE_addcard(Card (&h)[ACEHAND], Card c) {
  h[c&7]+=c;
  h[3]|=c;
}

static inline int ACE_rank(Card r) {
  return ((r)>>28);
}

Card ACE_evaluate(Card const (&h)[ACEHAND]);

} // namespace ace
