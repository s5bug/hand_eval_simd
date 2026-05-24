# hand_eval_simd

## background

"Video Poker" is a style of game where
- you are given 5 cards from a 52 card deck
- you have to choose a set of cards to discard from your hand
- those cards are replaced by a random choice from the remaining 47 cards
- this new hand is scored with High Card being negative and Royal Flush being ~250 points

The goal is to find what the optimal discards are given a specific hand.

I was “nerd-sniped” by Shakkar23: https://github.com/shakkar23/blackjack/commit/3f67dc86cd60f1def7be642bd04871d271a2e1cb
originally took ~1300ms to evaluate the best option for a given hand (multiple days for all possible hands). I helped
optimize this down to 600ms with some basic fixes (avoid division when possible), and then further to 400ms using
AVX-512 to evaluate the kind of hand a set of 5 cards is. Then, down to 130ms by loading the hand to evaluate directly
into a Struct-of-Arrays.

Then I had the idea of optimizing how the cards are actually sampled: the original code produced a vector of all
possible “replaced hands” given an incomplete hand. I figured this would be more efficient if a collection was never
built in the first place. Streaming hands directly into evaluation dropped the per-hand runtime down to 2ms, taking the
total time to build all results from multiple days to only a few minutes.

Iʼve published the code mostly for my own future reference, when I encounter a problem in the future that requires use
of SIMD.

## compilation

Itʼs best to use the Intel compiler (icx, icpx). The project can be built with Meson. Use a release buildtype and LTO
for the best results.
