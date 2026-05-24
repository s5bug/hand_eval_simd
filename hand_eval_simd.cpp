#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include <simde/x86/avx512/add.h>
#include <simde/x86/avx512/and.h>
#include <simde/x86/avx512/andnot.h>
#include <simde/x86/avx512/blend.h>
#include <simde/x86/avx512/cmpeq.h>
#include <simde/x86/avx512/loadu.h>
#include <simde/x86/avx512/lzcnt.h>
#include <simde/x86/avx512/or.h>
#include <simde/x86/avx512/popcnt.h>
#include <simde/x86/avx512/reduce.h>
#include <simde/x86/avx512/set1.h>
#include <simde/x86/avx512/slli.h>
#include <simde/x86/avx512/srlv.h>
#include <simde/x86/avx512/sub.h>

#define PROJECT_NAME "hand_eval_simd"

// rank stored as bitmask, but with enough bits to also store counts!
enum PokerRank : std::uint64_t {
    ACE   = 0x0000000000001,
    TWO   = 0x0000000000010,
    THREE = 0x0000000000100,
    FOUR  = 0x0000000001000,
    FIVE  = 0x0000000010000,
    SIX   = 0x0000000100000,
    SEVEN = 0x0000001000000,
    EIGHT = 0x0000010000000,
    NINE  = 0x0000100000000,
    TEN   = 0x0001000000000,
    JACK  = 0x0010000000000,
    QUEEN = 0x0100000000000,
    KING  = 0x1000000000000
};

constexpr inline const char* rank_symbol(const PokerRank rank) {
    switch (rank) {
        case ACE: return "a";
        case TWO: return "2";
        case THREE: return "3";
        case FOUR: return "4";
        case FIVE: return "5";
        case SIX: return "6";
        case SEVEN: return "7";
        case EIGHT: return "8";
        case NINE: return "9";
        case TEN: return "0";
        case JACK: return "j";
        case QUEEN: return "q";
        case KING: return "k";
        default: std::unreachable();
    }
}

// suit stored as bitmask
enum PokerSuit : std::uint64_t {
    SPADES = 0b0001,
    HEARTS = 0b0010,
    CLUBS  = 0b0100,
    DIAMONDS = 0b1000
};

constexpr inline const char* suit_symbol(const PokerSuit suit) {
    switch (suit) {
        case SPADES: return "S";
        case HEARTS: return "H";
        case CLUBS: return "C";
        case DIAMONDS: return "D";
        default: std::unreachable();
    }
}

struct Card {
    PokerRank rank;
    PokerSuit suit;
};

struct HandBatch;

struct HandReference {
    HandBatch& batch;
    std::size_t idx;

    constexpr inline PokerRank& rank(const std::size_t card_idx) const;
    constexpr inline PokerSuit& suit(const std::size_t card_idx) const;

    constexpr inline const HandReference& operator=(const std::array<Card, 5>& cards) const;
};

struct HandBatch {
    alignas(64) std::array<std::array<PokerRank, 8>, 5> ranks;
    alignas(64) std::array<std::array<PokerSuit, 8>, 5> suits;

    constexpr inline HandReference hand(const std::size_t idx) {
        return { *this, idx };
    }
};

constexpr inline PokerRank& HandReference::rank(const std::size_t card_idx) const {
    return batch.ranks[card_idx][idx];
}

constexpr inline PokerSuit& HandReference::suit(const std::size_t card_idx) const {
    return batch.suits[card_idx][idx];
}

constexpr inline const HandReference& HandReference::operator=(const std::array<Card, 5> &cards) const {
    for (std::size_t i = 0; i < 5; i++) {
        rank(i) = cards[i].rank;
        suit(i) = cards[i].suit;
    }

    return *this;
}

enum PokerHand : std::int64_t {
    HIGH_CARD = -1,
    PAIR = 0,
    TWO_PAIR = 1,
    THREE_OF_A_KIND = 2,
    STRAIGHT = 3,
    FLUSH = 4,
    FULL_HOUSE = 7,
    FOUR_OF_A_KIND = 24,
    STRAIGHT_FLUSH = 39,
    ROYAL_FLUSH = 249,
};

const char* hand_name(const PokerHand hand) {
    switch (hand) {
        case HIGH_CARD: return "High Card";
        case PAIR: return "Pair";
        case TWO_PAIR: return "Two Pair";
        case THREE_OF_A_KIND: return "Three of a Kind";
        case STRAIGHT: return "Straight";
        case FLUSH: return "Flush";
        case FULL_HOUSE: return "Full House";
        case FOUR_OF_A_KIND: return "Four of a Kind";
        case STRAIGHT_FLUSH: return "Straight Flush";
        case ROYAL_FLUSH: return "Royal Flush";
        default: std::unreachable();
    }
}

// __attribute__((optimize("O3")))
inline __attribute__((always_inline))
simde__m512i evaluate_hands_avx512(
    // mask of which suits have been seen
    const simde__m512i v_suit,
    // mask of which ranks have been seen
    const simde__m512i v_rank,
    // counts of which ranks have been seen (64 bit lane / 3 bit sections [0 - 4 is 3 bits] = 21 possible rank storage)
    const simde__m512i v_counts
) {
    // Constants
    const simde__m512i v_one = simde_mm512_set1_epi64(1);
    const simde__m512i v_two = simde_mm512_set1_epi64(2);
    const simde__m512i v_three = simde_mm512_set1_epi64(3);
    const simde__m512i v_four = simde_mm512_set1_epi64(4);
    const simde__m512i v_mask_0F = simde_mm512_set1_epi64(0b1111); // 1111 is the suit mask for a flush
    const simde__m512i v_31 = simde_mm512_set1_epi64(0x1'1111); // 0x1 1111 is the bit pattern for a regular straight
    const simde__m512i v_broadway = simde_mm512_set1_epi64(0x1'1110'0000'0001); // 0x1 1110 0000 0001 represents a broadway straight
    const simde__m512i v_63 = simde_mm512_set1_epi64(63);

    // --- 1. Flush Check ---
    // If the suit mask only has one bit set, the hand is a flush
    const simde__m512i v_suit_popcnt = simde_mm512_popcnt_epi64(v_suit);
    const simde__mmask8 is_flush = simde_mm512_cmpeq_epi64_mask(v_suit_popcnt, v_one);

    // --- 2. Straight Check ---
    // if (rankbits >> ctz(rankbits)) == 0x11111 = 31 we have a normal straight
    // _mm512_tzcnt_epi64 doesn't exist, so we take advantage of an equivalence
    // tzcnt(x) = popcnt(~x & (x - 1))
    const simde__m512i v_rank_minus_one = simde_mm512_sub_epi64(v_rank, v_one);
    const simde__m512i v_tz_mask = simde_mm512_andnot_si512(v_rank, v_rank_minus_one);
    const simde__m512i v_ctz = simde_mm512_popcnt_epi64(v_tz_mask);
    const simde__m512i v_shifted_ranks = simde_mm512_srlv_epi64(v_rank, v_ctz);

    const simde__mmask8 is_normal_straight = simde_mm512_cmpeq_epi64_mask(v_shifted_ranks, v_31);
    const simde__mmask8 is_broadway_straight = simde_mm512_cmpeq_epi64_mask(v_rank, v_broadway);
    const simde__mmask8 is_straight = is_normal_straight | is_broadway_straight;

    // --- 3. Rank Popcnt & Count Extraction ---
    // we know based on the popcnt of the rank mask the possible hands:
    // - it literally can't be 1 because that's 5oak
    // - if it's 2, that means either 4oak or full house
    // - if it's 3, that means either 3oak or 2pair
    // - if it's 4, that means pair
    // - if it's 5, that means straight or high card
    const simde__m512i v_rank_popcnt = simde_mm512_popcnt_epi64(v_rank);

    // Extract count at CTZ: (counts >> ctz) & 0xF
    // if the popcnt from earlier is:
    // - 2, then we either have a 4/1 split (4oak) or a 3/2 split (full house)
    // - 3, then we either have a 3/1/1 split (3oak) or a 2/2/1 split (2pair)
    // this means the count of the rank at the CTZ uniquely determines the hand for 2 unique ranks
    // for 3 unique ranks, if we see a count of 3 or a count of 2 it is also uniquely determined
    const simde__m512i v_shift_ctz = simde_mm512_popcnt_epi64(v_tz_mask);
    const simde__m512i v_count_ctz = simde_mm512_and_si512(
        simde_mm512_srlv_epi64(v_counts, v_shift_ctz), v_mask_0F
    );

    // Extract count at CLZ: (counts >> (63 - clz)) & 0xF
    // if we see a count of 1 at the CTZ, then we should check the CLZ
    // if we see 1/2 we know it's 2pair, if we see 1/3 we know it's 3oak
    const simde__m512i v_clz = simde_mm512_lzcnt_epi64(v_rank);
    const simde__m512i v_shift_clz = simde_mm512_sub_epi64(v_63, v_clz);
    const simde__m512i v_count_high = simde_mm512_and_si512(
        simde_mm512_srlv_epi64(v_counts, v_shift_clz), v_mask_0F
    );

    // --- 4. Evaluate Combinations ---

    // Popcnt == 2 (4OAK or Full House)
    const simde__mmask8 is_pop2 = simde_mm512_cmpeq_epi64_mask(v_rank_popcnt, v_two);
    const simde__mmask8 is_ctz_1 = simde_mm512_cmpeq_epi64_mask(v_count_ctz, v_one);
    const simde__mmask8 is_ctz_4 = simde_mm512_cmpeq_epi64_mask(v_count_ctz, v_four);
    const simde__mmask8 is_4oak = is_pop2 & (is_ctz_1 | is_ctz_4);
    const simde__mmask8 is_full_house = is_pop2 & ~is_4oak;

    // Popcnt == 3 (3OAK or 2Pair)
    const simde__mmask8 is_pop3 = simde_mm512_cmpeq_epi64_mask(v_rank_popcnt, v_three);
    const simde__mmask8 is_ctz_3 = simde_mm512_cmpeq_epi64_mask(v_count_ctz, v_three);
    const simde__mmask8 is_ctz_2 = simde_mm512_cmpeq_epi64_mask(v_count_ctz, v_two);
    const simde__mmask8 is_high_2 = simde_mm512_cmpeq_epi64_mask(v_count_high, v_two);

    const simde__mmask8 is_pop3_ctz1 = is_pop3 & is_ctz_1;
    const simde__mmask8 is_3oak = (is_pop3 & is_ctz_3) | (is_pop3_ctz1 & ~is_high_2);
    const simde__mmask8 is_2pair = (is_pop3 & is_ctz_2) | (is_pop3_ctz1 & is_high_2);

    // Popcnt == 4 (Pair)
    const simde__mmask8 is_pair = simde_mm512_cmpeq_epi64_mask(v_rank_popcnt, v_four);

    // --- 5. Resolve Final Ranks ---
    // Start with HIGH_CARD baseline, layer masks on top to override.
    // Masks are mutually exclusive among non-straight/flush hands.
    simde__m512i v_result = simde_mm512_set1_epi64(HIGH_CARD);

    v_result = simde_mm512_mask_blend_epi64(is_pair, v_result, simde_mm512_set1_epi64(PAIR));
    v_result = simde_mm512_mask_blend_epi64(is_2pair, v_result, simde_mm512_set1_epi64(TWO_PAIR));
    v_result = simde_mm512_mask_blend_epi64(is_3oak, v_result, simde_mm512_set1_epi64(THREE_OF_A_KIND));
    v_result = simde_mm512_mask_blend_epi64(is_straight, v_result, simde_mm512_set1_epi64(STRAIGHT));
    v_result = simde_mm512_mask_blend_epi64(is_flush, v_result, simde_mm512_set1_epi64(FLUSH));
    v_result = simde_mm512_mask_blend_epi64(is_full_house, v_result, simde_mm512_set1_epi64(FULL_HOUSE));
    v_result = simde_mm512_mask_blend_epi64(is_4oak, v_result, simde_mm512_set1_epi64(FOUR_OF_A_KIND));

    // Straight Flush overwrites standalone straight or flush masks
    v_result = simde_mm512_mask_blend_epi64(is_straight & is_flush, v_result, simde_mm512_set1_epi64(STRAIGHT_FLUSH));

    // Royal Flush overrides Straight Flush
    v_result = simde_mm512_mask_blend_epi64(is_broadway_straight & is_flush, v_result, simde_mm512_set1_epi64(ROYAL_FLUSH));

    return v_result;
}

inline __attribute__((always_inline))
simde__m512i evaluate_batch(
    const HandBatch& batch
    ) {
    simde__m512i v_suit_mask = simde_mm512_setzero_si512();
    simde__m512i v_rank_mask = simde_mm512_setzero_si512();
    simde__m512i v_rank_counts = simde_mm512_setzero_si512();

    for (std::size_t card = 0; card < 5; ++card) {
        const simde__m512i v_ranks = simde_mm512_load_si512(batch.ranks[card].data());
        const simde__m512i v_suits = simde_mm512_load_si512(batch.suits[card].data());

        v_suit_mask = simde_mm512_or_si512(v_suit_mask, v_suits);
        v_rank_mask = simde_mm512_or_si512(v_rank_mask, v_ranks);
        v_rank_counts = simde_mm512_add_epi64(v_rank_counts, v_ranks);
    }

    return evaluate_hands_avx512(v_suit_mask, v_rank_mask, v_rank_counts);
}

std::int64_t evaluate_batches(
    const std::size_t num_hands,
    const HandBatch* batches
) {
    const std::size_t num_batches = num_hands / 8;
    const std::size_t num_remainder = num_hands % 8;

    simde__m512i total_score = simde_mm512_setzero_si512();
    for(std::size_t i = 0; i < num_batches; i++) {
        const simde__m512i batch_score = evaluate_batch(batches[i]);
        total_score = simde_mm512_add_epi64(total_score, batch_score);
    }

    if (num_remainder > 0) {
        const simde__m512i batch_score = evaluate_batch(batches[num_batches]);

        // Create a bitmask for the valid hands (e.g., a remainder of 3 creates 0b00000111)
        const simde__mmask8 valid_mask = (1 << num_remainder) - 1;
        total_score = simde_mm512_mask_add_epi64(total_score, valid_mask, total_score, batch_score);
    }

    return simde_mm512_reduce_add_epi64(total_score);
}

struct RemainingDeck {
    std::array<Card, 47> cards;

    static constexpr uint64_t all_cards = (1ULL << 52) - 1;
};

constexpr inline Card id_to_card(const std::uint64_t id) {
    return {
        // (id & ~3) uses all bits but the last two (i.e. the 4n in 4n + suit) and shifts to get the rank enum value
        static_cast<PokerRank>(1ULL << (id & ~0b11)),
        // (id & 3) uses the last two bits as a [0, 3] index for the suit bit
        static_cast<PokerSuit>(1ULL << (id & 0b11))
    };
}

constexpr inline std::uint64_t card_to_mask(const Card c) {
    // a u64 has >52 bits
    // we can take the rank and shift it by the ctz of the suit to get a unique bit for a card
    const std::uint64_t suitIdx = std::countr_zero(static_cast<std::uint64_t>(c.suit));
    return c.rank << suitIdx;
}

constexpr inline RemainingDeck mask47_to_deck(std::uint64_t mask) {
    // PRECONDITION: popcnt(mask) == (52 - 5)
    RemainingDeck deck;
    std::size_t idx = 0;
    while (mask != 0) {
        const int card_id = std::countr_zero(mask);
        deck.cards[idx++] = id_to_card(card_id);

        // clear the lowest bit of mask to process the next card
        mask &= (mask - 1);
    }
    return deck;
}

constexpr inline RemainingDeck remaining_deck_for_hand(const std::array<Card, 5>& hand) {
    std::uint64_t mask = RemainingDeck::all_cards;
    for (const auto& c : hand) {
        mask &= ~(card_to_mask(c));
    }
    return mask47_to_deck(mask);
}

template<std::size_t cards_to_pull, std::invocable<std::array<Card, cards_to_pull>> Consumer>
requires(cards_to_pull <= 5)
inline void generate_pulls(const RemainingDeck& deck, Consumer&& cb) {
    if constexpr (cards_to_pull == 0) {
        cb(std::array<Card, 0>{});
    } else if constexpr (cards_to_pull == 1) {
        for (std::size_t i1 = 0; i1 < 47; i1++) {
            cb(std::array<Card, 1>{deck.cards[i1]});
        }
    } else if constexpr (cards_to_pull == 2) {
        for (std::size_t i1 = 0; i1 < 46; i1++) {
            for (std::size_t i2 = 1 + i1; i2 < 47; i2++) {
                cb(std::array<Card, 2>{deck.cards[i1], deck.cards[i2]});
            }
        }
    } else if constexpr (cards_to_pull == 3) {
        for (std::size_t i1 = 0; i1 < 45; i1++) {
            for (std::size_t i2 = 1 + i1; i2 < 46; i2++) {
                for (std::size_t i3 = 1 + i2; i3 < 47; i3++) {
                    cb(std::array<Card, 3>{deck.cards[i1], deck.cards[i2], deck.cards[i3]});
                }
            }
        }
    } else if constexpr (cards_to_pull == 4) {
        for (std::size_t i1 = 0; i1 < 44; i1++) {
            for (std::size_t i2 = 1 + i1; i2 < 45; i2++) {
                for (std::size_t i3 = 1 + i2; i3 < 46; i3++) {
                    for (std::size_t i4 = 1 + i3; i4 < 47; i4++) {
                        cb(std::array<Card, 4>{deck.cards[i1], deck.cards[i2], deck.cards[i3], deck.cards[i4]});
                    }
                }
            }
        }
    } else if constexpr (cards_to_pull == 5) {
        for (std::size_t i1 = 0; i1 < 43; i1++) {
            for (std::size_t i2 = 1 + i1; i2 < 44; i2++) {
                for (std::size_t i3 = 1 + i2; i3 < 45; i3++) {
                    for (std::size_t i4 = 1 + i3; i4 < 46; i4++) {
                        for (std::size_t i5 = 1 + i4; i5 < 47; i5++) {
                            cb(std::array<Card, 5>{deck.cards[i1], deck.cards[i2], deck.cards[i3], deck.cards[i4], deck.cards[i5]});
                        }
                    }
                }
            }
        }
    } else static_assert(false, "cards_to_pull must be ∈ [0..5]");
}

template<std::size_t cards_discarded>
std::int64_t score_of_hand_with(const RemainingDeck& deck, const std::array<Card, 5 - cards_discarded> hand_after_discard) {
    HandBatch current_batch;

    for (std::size_t card_idx = cards_discarded; card_idx < 5; ++card_idx) {
        const Card& c = hand_after_discard[card_idx - cards_discarded];
        for (std::size_t hand_idx = 0; hand_idx < 8; hand_idx++) {
            current_batch.hand(hand_idx).rank(card_idx) = c.rank;
            current_batch.hand(hand_idx).suit(card_idx) = c.suit;
        }
    }

    std::size_t hand_idx = 0;
    simde__m512i total_score = simde_mm512_setzero_si512();

    generate_pulls<cards_discarded>(deck, [&](const auto& pulled_cards) {
        for (std::size_t i = 0; i < cards_discarded; i++) {
            const Card& c = pulled_cards[i];
            current_batch.hand(hand_idx).rank(i) = c.rank;
            current_batch.hand(hand_idx).suit(i) = c.suit;
        }

        hand_idx++;
        if (hand_idx == 8) {
            total_score = simde_mm512_add_epi64(total_score, evaluate_batch(current_batch));
            hand_idx = 0;
        }
    });

    if (hand_idx > 0) {
        const simde__m512i batch_score = evaluate_batch(current_batch);

        // Create a bitmask for the valid hands (e.g., a remainder of 3 creates 0b00000111)
        const simde__mmask8 valid_mask = (1 << hand_idx) - 1;
        total_score = simde_mm512_mask_add_epi64(total_score, valid_mask, total_score, batch_score);
    }

    return simde_mm512_reduce_add_epi64(total_score);
}

// 00000 = discarding nothing
// 11111 = discarding everything
// so that's 32 possible discard choices
using ScoreAfterDiscards = std::array<std::int64_t, 32>;

template <std::size_t num_kept>
inline std::array<Card, num_kept> get_kept_cards(const std::array<Card, 5>& hand, const std::size_t discard_mask) {
    std::array<Card, num_kept> kept;
    std::size_t idx = 0;
    for (std::size_t i = 0; i < 5; ++i) {
        // if the bit at position i is 0, the card is kept
        if ((discard_mask & (1 << i)) == 0) {
            kept[idx++] = hand[i];
        }
    }
    return kept;
}

constexpr inline std::array<Card, 5> mask_to_hand5(std::uint64_t mask) {
    std::array<Card, 5> hand;
    hand[0] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[1] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[2] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[3] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[4] = id_to_card(std::countr_zero(mask));
    return hand;
}

constexpr inline ScoreAfterDiscards all_scores_for(const std::uint64_t hand_bits) {
    const std::array<Card, 5> hand = mask_to_hand5(hand_bits);
    const std::uint64_t deck_mask = RemainingDeck::all_cards & ~hand_bits;
    const RemainingDeck deck = mask47_to_deck(deck_mask);

    ScoreAfterDiscards scores;

    for (std::size_t discard_mask = 0; discard_mask < 0b11111; discard_mask++) {
        const std::size_t num_discarded = std::popcount(discard_mask);

        switch (num_discarded) {
            case 0:
                scores[discard_mask] = score_of_hand_with<0>(deck, get_kept_cards<5>(hand, discard_mask));
                break;
            case 1:
                scores[discard_mask] = score_of_hand_with<1>(deck, get_kept_cards<4>(hand, discard_mask));
                break;
            case 2:
                scores[discard_mask] = score_of_hand_with<2>(deck, get_kept_cards<3>(hand, discard_mask));
                break;
            case 3:
                scores[discard_mask] = score_of_hand_with<3>(deck, get_kept_cards<2>(hand, discard_mask));
                break;
            case 4:
                scores[discard_mask] = score_of_hand_with<4>(deck, get_kept_cards<1>(hand, discard_mask));
                break;
            case 5:
                scores[discard_mask] = score_of_hand_with<5>(deck, get_kept_cards<0>(hand, discard_mask));
                break;
            default: std::unreachable();
        }
    }

    return scores;
}

template<std::uint64_t n, std::uint64_t k>
std::uint64_t n_choose_k = ([]() constexpr -> std::uint64_t {
    std::uint64_t kr = k;

    if (kr < 0 || kr > n) return 0;
    if (kr == 0 || kr == n) return 1;
    if (kr > (n / 2)) kr = n - kr;

    long long res = 1;
    for (std::uint64_t i = 1; i <= kr; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
})();

constexpr inline std::uint64_t denominator(std::uint64_t mask) {
    switch (std::popcount(mask)) {
        case 0: return n_choose_k<47, 0>;
        case 1: return n_choose_k<47, 1>;
        case 2: return n_choose_k<47, 2>;
        case 3: return n_choose_k<47, 3>;
        case 4: return n_choose_k<47, 4>;
        case 5: return n_choose_k<47, 5>;
        default: std::unreachable();
    }
}

int main(int argc, char **argv) {
    if (argc != 1) {
        std::cout << argv[0] << " takes no arguments.\n";
        return 1;
    }
    std::cout << "This is project " << PROJECT_NAME << ".\n";

    std::ofstream outfile("results.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open results.txt\n";
        return 1;
    }

    constexpr int total_chunks = (48 * 49) / 2;
    std::atomic<std::size_t> iterations_complete {0};

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::uint64_t ci = 0; ci < 48; ci++) {
        for (std::uint64_t cj = 1 + ci; cj < 49; cj++) {
            const std::uint64_t c1 = 1ULL << ci;
            const std::uint64_t c2 = 1ULL << cj;

            std::ostringstream local_buffer;

            for (std::uint64_t c3 = c2 << 1; c3 < (1ULL << (52 - 2)); c3 <<= 1) {
                for (std::uint64_t c4 = c3 << 1; c4 < (1ULL << (52 - 1)); c4 <<= 1) {
                    for (std::uint64_t c5 = c4 << 1; c5 < (1ULL << (52 - 0)); c5 <<= 1) {
                        auto before = std::chrono::system_clock::now();

                        const std::uint64_t hand = c1 | c2 | c3 | c4 | c5;
                        const ScoreAfterDiscards scores = all_scores_for(hand);

                        std::uint64_t best_mask = 0;
                        std::int64_t best_score = std::numeric_limits<std::int64_t>::min();
                        std::uint64_t best_denominator = 1;
                        for (std::size_t i = 0; i < 0b11111; i++) {
                            const std::uint64_t current_denom = denominator(i);
                            const std::int64_t other_num = current_denom * best_score;
                            const std::int64_t current_num = best_denominator * scores[i];

                            if (current_num > other_num) {
                                best_mask = i;
                                best_score = scores[i];
                                best_denominator = current_denom;
                            }
                        }

                        auto after = std::chrono::system_clock::now();

                        std::array<Card, 5> cards = mask_to_hand5(hand);
                        for (std::size_t i = 0; i < 5; i++) {
                            const char* rank = rank_symbol(cards[i].rank);
                            const char* suit = suit_symbol(cards[i].suit);

                            local_buffer << rank << suit;
                            if (i != 4) local_buffer << " ";
                            else local_buffer << ": ";
                        }

                        std::string bit_mask_text = std::format("{:0>5b}", best_mask);
                        std::ranges::reverse(bit_mask_text);
                        local_buffer << bit_mask_text << " " << best_score << "/" << best_denominator << ", ";
                        local_buffer << std::chrono::duration<double, std::milli>(after - before).count() << "\n";
                    }
                }
            }

            #pragma omp critical
            {
                outfile << local_buffer.str();
                ++iterations_complete;
                std::cout << "iteration " << iterations_complete << "/" << total_chunks << " complete" << std::endl;
            }
        }
    }

    return 0;
}
