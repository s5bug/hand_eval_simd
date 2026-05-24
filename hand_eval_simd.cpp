#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>

#include <simde/x86/avx512/add.h>
#include <simde/x86/avx512/and.h>
#include <simde/x86/avx512/andnot.h>
#include <simde/x86/avx512/blend.h>
#include <simde/x86/avx512/cmpeq.h>
#include <simde/x86/avx512/cmpneq.h>
#include <simde/x86/avx512/loadu.h>
#include <simde/x86/avx512/lzcnt.h>
#include <simde/x86/avx512/or.h>
#include <simde/x86/avx512/popcnt.h>
#include <simde/x86/avx512/reduce.h>
#include <simde/x86/avx512/set1.h>
#include <simde/x86/avx512/slli.h>
#include <simde/x86/avx512/srlv.h>
#include <simde/x86/avx512/store.h>
#include <simde/x86/avx512/sub.h>

#include "./concurrentqueue.h"
#include "./blockingconcurrentqueue.h"

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

constexpr inline char rank_symbol(const PokerRank rank) {
    switch (rank) {
        case ACE: return 'a';
        case TWO: return '2';
        case THREE: return '3';
        case FOUR: return '4';
        case FIVE: return '5';
        case SIX: return '6';
        case SEVEN: return '7';
        case EIGHT: return '8';
        case NINE: return '9';
        case TEN: return '0';
        case JACK: return 'j';
        case QUEEN: return 'q';
        case KING: return 'k';
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

constexpr inline char suit_symbol(const PokerSuit suit) {
    switch (suit) {
        case SPADES: return 'S';
        case HEARTS: return 'H';
        case CLUBS: return 'C';
        case DIAMONDS: return 'D';
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

template<typename T>
concept Int64Enum = std::is_enum_v<T> &&
                     std::is_same_v<std::underlying_type_t<T>, std::int64_t>;

enum class PokerHandMobileApp : std::int64_t {
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
static_assert(Int64Enum<PokerHandMobileApp>);

enum class PokerHandJacksOrBetter : std::int64_t {
    OTHER = -1,
    JACKS_OR_BETTER = 0,
    TWO_PAIR = 1,
    THREE_OF_A_KIND = 2,
    STRAIGHT = 3,
    FLUSH = 5,
    FULL_HOUSE = 8,
    FOUR_OF_A_KIND = 24,
    STRAIGHT_FLUSH = 49,
    ROYAL_FLUSH = 799
};

template<Int64Enum Hand>
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

    simde__m512i v_result;

    // --- 5. Resolve Final Ranks ---
    // Start with HIGH_CARD baseline, layer masks on top to override.
    // Masks are mutually exclusive among non-straight/flush hands.
    if constexpr (std::same_as<Hand, PokerHandMobileApp>) {
        using enum PokerHandMobileApp;
        v_result = simde_mm512_set1_epi64(static_cast<int64_t>(HIGH_CARD));

        v_result = simde_mm512_mask_blend_epi64(is_pair, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(PAIR)));
        v_result = simde_mm512_mask_blend_epi64(is_2pair, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(TWO_PAIR)));
        v_result = simde_mm512_mask_blend_epi64(is_3oak, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(THREE_OF_A_KIND)));
        v_result = simde_mm512_mask_blend_epi64(is_straight, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(STRAIGHT)));
        v_result = simde_mm512_mask_blend_epi64(is_flush, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(FLUSH)));
        v_result = simde_mm512_mask_blend_epi64(is_full_house, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(FULL_HOUSE)));
        v_result = simde_mm512_mask_blend_epi64(is_4oak, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(FOUR_OF_A_KIND)));

        // Straight Flush overwrites standalone straight or flush masks
        v_result = simde_mm512_mask_blend_epi64(is_straight & is_flush, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(STRAIGHT_FLUSH)));

        // Royal Flush overrides Straight Flush
        v_result = simde_mm512_mask_blend_epi64(is_broadway_straight & is_flush, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(ROYAL_FLUSH)));
    } else if constexpr (std::same_as<Hand, PokerHandJacksOrBetter>) {
        using enum PokerHandJacksOrBetter;
        v_result = simde_mm512_set1_epi64(static_cast<int64_t>(OTHER));

        const simde__m512i v_jacks_or_better = simde_mm512_set1_epi64(0x2'2200'0000'0002); // a count of 2 in any JKQA slot
        // Need to check if the pair is in JKQA range
        const simde__m512i is_jack_pair_bits = simde_mm512_and_si512(v_counts, v_jacks_or_better);
        const simde__mmask8 is_jack_pair_mask = simde_mm512_cmpneq_epi64_mask(is_jack_pair_bits, simde_mm512_setzero_si512());

        v_result = simde_mm512_mask_blend_epi64(is_pair & is_jack_pair_mask, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(JACKS_OR_BETTER)));
        v_result = simde_mm512_mask_blend_epi64(is_2pair, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(TWO_PAIR)));
        v_result = simde_mm512_mask_blend_epi64(is_3oak, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(THREE_OF_A_KIND)));
        v_result = simde_mm512_mask_blend_epi64(is_straight, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(STRAIGHT)));
        v_result = simde_mm512_mask_blend_epi64(is_flush, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(FLUSH)));
        v_result = simde_mm512_mask_blend_epi64(is_full_house, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(FULL_HOUSE)));
        v_result = simde_mm512_mask_blend_epi64(is_4oak, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(FOUR_OF_A_KIND)));

        // Straight Flush overwrites standalone straight or flush masks
        v_result = simde_mm512_mask_blend_epi64(is_straight & is_flush, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(STRAIGHT_FLUSH)));

        // Royal Flush overrides Straight Flush
        v_result = simde_mm512_mask_blend_epi64(is_broadway_straight & is_flush, v_result, simde_mm512_set1_epi64(static_cast<int64_t>(ROYAL_FLUSH)));
    } else {
        static_assert(Int64Enum<Hand> && false, "these rules are not implemented");
        std::unreachable();
    }

    return v_result;
}

template<Int64Enum Hand>
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

    return evaluate_hands_avx512<Hand>(v_suit_mask, v_rank_mask, v_rank_counts);
}

template<Int64Enum Hand>
std::int64_t evaluate_batches(
    const std::size_t num_hands,
    const HandBatch* batches
) {
    const std::size_t num_batches = num_hands / 8;
    const std::size_t num_remainder = num_hands % 8;

    simde__m512i total_score = simde_mm512_setzero_si512();
    for(std::size_t i = 0; i < num_batches; i++) {
        const simde__m512i batch_score = evaluate_batch<Hand>(batches[i]);
        total_score = simde_mm512_add_epi64(total_score, batch_score);
    }

    if (num_remainder > 0) {
        const simde__m512i batch_score = evaluate_batch<Hand>(batches[num_batches]);

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

template<Int64Enum Hand, std::size_t cards_discarded>
std::int64_t score_of_hand_with(const RemainingDeck& deck, const HandBatch& base_batch, const std::array<size_t, 5>& discard_slots) {
    HandBatch current_batch = base_batch;

    std::size_t hand_idx = 0;
    simde__m512i total_score = simde_mm512_setzero_si512();

    generate_pulls<cards_discarded>(deck, [&](const auto& pulled_cards) {
        for (std::size_t i = 0; i < cards_discarded; i++) {
            const Card& c = pulled_cards[i];
            const std::size_t slot = discard_slots[i];

            current_batch.hand(hand_idx).rank(slot) = c.rank;
            current_batch.hand(hand_idx).suit(slot) = c.suit;
        }

        hand_idx++;
        if (hand_idx == 8) {
            total_score = simde_mm512_add_epi64(total_score, evaluate_batch<Hand>(current_batch));
            hand_idx = 0;
        }
    });

    if (hand_idx > 0) {
        const simde__m512i batch_score = evaluate_batch<Hand>(current_batch);

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

constexpr inline std::array<Card, 5> mask_to_hand5(std::uint64_t mask) {
    std::array<Card, 5> hand;
    hand[0] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[1] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[2] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[3] = id_to_card(std::countr_zero(mask)); mask &= (mask - 1);
    hand[4] = id_to_card(std::countr_zero(mask));
    return hand;
}

constexpr std::array<std::array<std::size_t, 5>, 32> DISCARD_SLOTS = ([]() constexpr {
    std::array<std::array<std::size_t, 5>, 32> arr{};
    for (std::size_t mask = 0; mask < 32; ++mask) {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < 5; ++i) {
            if (mask & (1 << i)) arr[mask][idx++] = i;
        }
    }
    return arr;
})();

template<Int64Enum Hand>
constexpr inline ScoreAfterDiscards all_scores_for(const std::uint64_t hand_bits, const std::array<Card, 5>& hand) {
    const std::uint64_t deck_mask = RemainingDeck::all_cards & ~hand_bits;
    const RemainingDeck deck = mask47_to_deck(deck_mask);

    HandBatch base_batch;
    for (std::size_t i = 0; i < 5; ++i) {
        simde_mm512_store_si512(base_batch.ranks[i].data(), simde_mm512_set1_epi64(hand[i].rank));
        simde_mm512_store_si512(base_batch.suits[i].data(), simde_mm512_set1_epi64(hand[i].suit));
    }

    ScoreAfterDiscards scores;

    for (std::size_t discard_mask = 0; discard_mask <= 0b11111; discard_mask++) {
        const std::size_t num_discarded = std::popcount(discard_mask);

        const auto& discard_slots = DISCARD_SLOTS[discard_mask];

        switch (num_discarded) {
            case 0:
                scores[discard_mask] = score_of_hand_with<Hand, 0>(deck, base_batch, discard_slots);
                break;
            case 1:
                scores[discard_mask] = score_of_hand_with<Hand, 1>(deck, base_batch, discard_slots);
                break;
            case 2:
                scores[discard_mask] = score_of_hand_with<Hand, 2>(deck, base_batch, discard_slots);
                break;
            case 3:
                scores[discard_mask] = score_of_hand_with<Hand, 3>(deck, base_batch, discard_slots);
                break;
            case 4:
                scores[discard_mask] = score_of_hand_with<Hand, 4>(deck, base_batch, discard_slots);
                break;
            case 5:
                scores[discard_mask] = score_of_hand_with<Hand, 5>(deck, base_batch, discard_slots);
                break;
            default: std::unreachable();
        }
    }

    return scores;
}

template<std::uint64_t n, std::uint64_t k>
constexpr std::uint64_t n_choose_k = ([]() constexpr -> std::uint64_t {
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

constexpr std::uint64_t num_canonical_hands = ([]() constexpr -> std::uint64_t {
    // Burnside's Lemma tells us how to reduce symmetry in counting groups
    // For each "actor" on our hands that preserves the structure, we can take "how many ways to apply that actor"
    // Add those all up and divide by the total number of actors

    // There is one way to have all hands
    constexpr std::uint64_t identity = n_choose_k<52, 5>;

    // There are six (4 choose 2) ways to swap two suits
    // If the two suits we choose:
    // - Have zero cards each in them, then there's 26c5 possibilities for the rest of the cards
    // - Have one card of the same rank each in them, then there's 13c1 choices for the pair rank and 26c3 choices for the rest of the cards
    // - Have two cards of the same rank each in them, then there's 13c2 choices for the pair ranks and 26c1 choices for the other card
    constexpr std::uint64_t single_swap =
        n_choose_k<26, 5> +
            (n_choose_k<13, 1> * n_choose_k<26, 3>) +
                (n_choose_k<13, 2> * n_choose_k<26, 1>);

    // There are three ways to swap two sets of suits
    // (4 choose 2 (the first pair) * 2 choose 2 (the second pair)) / 2! (permute the pairings) = 6/2 = 3
    // A hand has exactly 5 cards so there is no way we can do this without changing the meaning of a hand
    constexpr std::uint64_t double_swap = 0;

    // There are eight (2! * (4 choose 3)) ways to cycle three suits
    // If the three suits we choose:
    // - Have zero cards each in them, then there's 13c5 possibilities for the rest of the cards
    // - Have one card each in them, then there's 13c1 possibilities for the triplet rank, and 13c2 possibilities for the rest of the cards
    constexpr std::uint64_t three_cycle =
        n_choose_k<13, 5> +
            (n_choose_k<13, 1> * n_choose_k<13, 2>);

    // There are six (3! * (4 choose 4)) ways to cycle four suits
    // A hand has no more than 4 suits so there is no way we can do this without changing the meaning of a hand
    constexpr std::uint64_t four_cycle = 0;

    constexpr std::uint64_t ops =
        (1 * identity) +
            (6 * single_swap) +
                (3 * double_swap) +
                    (8 * three_cycle) +
                        (6 * four_cycle);

    // 4! ways to permute the suits
    constexpr std::uint64_t suit_permutation_count = 24;

    return ops / suit_permutation_count;
})();

constexpr std::array<std::array<std::uint64_t, 4>, 24> SUIT_PERMUTATIONS = ([]() constexpr {
    std::array<std::array<std::uint64_t, 4>, 24> perms{};
    // we permute the indices of the suits because in get_weight we do math on bit position
    std::array<std::uint64_t, 4> current = {0, 1, 2, 3};
    std::size_t idx = 0;
    do {
        perms[idx++] = current;
    } while (std::ranges::next_permutation(current).found);
    return perms;
})();

inline std::uint64_t get_weight(const std::uint64_t hand) {
    std::array<std::uint64_t, 5> card_indices {};

    std::uint64_t temp = hand;
    for (int i = 0; i < 5; ++i) {
        // get the lowest bit and clear it
        card_indices[i] = std::countr_zero(temp);
        temp &= temp - 1;
    }

    std::uint64_t unique_masks = 0;
    std::array<std::uint64_t, 24> seen_masks{};

    for (const auto& perm : SUIT_PERMUTATIONS) {
        std::uint64_t mapped_hand = 0;
        for (int i = 0; i < 5; ++i) {
            // each rank is every 4 bits, meaning the actual rank value is the index rounded down to the nearest 16
            const int rank_base = card_indices[i] & ~3;
            // the suit is the remainder of this rounding down
            const int original_suit = card_indices[i] & 3;
            const int mapped_suit = perm[original_suit];
            mapped_hand |= (1ULL << (rank_base + mapped_suit));
        }

        if (mapped_hand > hand) {
            // we know that this isn't the canonical hand
            // so it shouldn't appear in results
            return 0;
        }

        // we keep track of all unique masks to discover the weight value
        bool is_unique = true;
        for (std::uint64_t u = 0; u < unique_masks; ++u) {
            if (seen_masks[u] == mapped_hand) {
                is_unique = false;
                break;
            }
        }
        // if we haven't seen this permutation before, add it to our filter
        if (is_unique) {
            seen_masks[unique_masks++] = mapped_hand;
        }
    }

    // this is the canonical hand, so we return how many other hands it has absorbed
    return unique_masks;
}

// a hand unique up to suit symmetries
struct CanonicalHand {
    std::uint64_t mask;
    // how many times does this hand "occur"
    std::uint64_t weight;
};

std::vector<CanonicalHand> generate_canonical_hands() {
    std::vector<CanonicalHand> result;
    result.reserve(num_canonical_hands + 1);

    for (std::uint64_t c1 = 1; c1 < (1ULL << (52 - 4)); c1 <<= 1) {
        for (std::uint64_t c2 = c1 << 1; c2 < (1ULL << (52 - 3)); c2 <<= 1) {
            for (std::uint64_t c3 = c2 << 1; c3 < (1ULL << (52 - 2)); c3 <<= 1) {
                for (std::uint64_t c4 = c3 << 1; c4 < (1ULL << (52 - 1)); c4 <<= 1) {
                    for (std::uint64_t c5 = c4 << 1; c5 < (1ULL << (52 - 0)); c5 <<= 1) {
                        const std::uint64_t hand = c1 | c2 | c3 | c4 | c5;

                        const std::uint64_t weight = get_weight(hand);
                        if (weight != 0) {
                            result.push_back({hand, weight});
                        }
                    }
                }
            }
        }
    }
    return result;
}

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

constexpr std::array<const char*, 32> REVERSED_BITMASKS = {
    "00000", "10000", "01000", "11000", "00100", "10100", "01100", "11100",
    "00010", "10010", "01010", "11010", "00110", "10110", "01110", "11110",
    "00001", "10001", "01001", "11001", "00101", "10101", "01101", "11101",
    "00011", "10011", "01011", "11011", "00111", "10111", "01111", "11111"
};

constexpr std::array<std::uint64_t, 32> DENOMINATORS = ([]() constexpr -> std::array<std::uint64_t, 32> {
    std::array<std::uint64_t, 32> arr{};
    for (int i = 0; i < 32; ++i) arr[i] = denominator(i);
    return arr;
})();

struct HandResult {
    std::uint64_t original_hand;
    std::uint8_t mask;
    std::int64_t score;
    std::uint64_t weight;
};

moodycamel::BlockingConcurrentQueue<HandResult> write_queue;
std::atomic<bool> computations_done{false};

void background_writer(std::ofstream&& outfile) {
    moodycamel::ConsumerToken write_queue_consumption(write_queue);

    // we want the total EV of the game to be (sum(individual EVs))/(52 choose 5)
    constexpr std::uint64_t LCM_DENOMINATOR = ([]() constexpr -> std::uint64_t {
        return std::accumulate(DENOMINATORS.begin(), DENOMINATORS.end(), 1ULL,
            [](const std::uint64_t a, const std::uint64_t b) { return std::lcm(a, b); });
    })();

    // so for each EV, we can calculate what we need to multiply it by to make it the proper numerator
    constexpr std::array<std::uint64_t, 32> SCALARS = ([]() constexpr -> std::array<std::uint64_t, 32> {
        std::array<std::uint64_t, 32> arr{};
        for (int i = 0; i < 32; ++i) arr[i] = LCM_DENOMINATOR / denominator(i);
        return arr;
    })();

    std::int64_t total_ev_numerator = 0;

    std::string local_string;
    local_string.reserve(2048 * 64);
    std::array<char, std::numeric_limits<std::uint64_t>::digits10> num_buf;
    std::array<HandResult, 2048> buffer;

    std::size_t total_written = 0;
    const auto before = std::chrono::steady_clock::now();

    while (true) {
        std::size_t elements_received = 0;

        do {
            elements_received = write_queue.wait_dequeue_bulk_timed(write_queue_consumption, buffer.data(), buffer.size(), std::chrono::milliseconds(4000));

            if (elements_received == 0 && computations_done.load(std::memory_order_acquire)) {
                constexpr std::uint64_t TOTAL_STARTING_HANDS = n_choose_k<52, 5>;
                constexpr std::uint64_t FINAL_DENOMINATOR = LCM_DENOMINATOR * TOTAL_STARTING_HANDS;

                outfile << std::format("Total EV: {}/{}\n", total_ev_numerator, FINAL_DENOMINATOR) << std::endl;

                return;
            }
        } while (elements_received == 0);

        for (std::size_t i = 0; i < elements_received; i++) {
            const auto& result = buffer[i];

            // the total ev is (our score) * (correction term for LCM) * (number of permutations we ate)
            total_ev_numerator += result.score * SCALARS[result.mask] * result.weight;

            const std::array<Card, 5> cards = mask_to_hand5(result.original_hand);

            std::array<char, 16> hand_string;
            for (std::size_t ci = 0; ci < 5; ci++) {
                hand_string[ci * 3] = rank_symbol(cards[ci].rank);
                hand_string[ci * 3 + 1] = suit_symbol(cards[ci].suit);
                hand_string[ci * 3 + 2] = (ci != 4) ? ' ' : ':';
            }
            hand_string[15] = ' ';

            local_string.append(hand_string.data(), 16);

            local_string += REVERSED_BITMASKS[result.mask];
            local_string += " ";

            auto [ptr0, ec0] = std::to_chars(num_buf.data(), num_buf.data() + num_buf.size(), result.weight);
            local_string.append(num_buf.data(), ptr0 - num_buf.data());
            local_string += "x ";

            auto [ptr1, ec1] = std::to_chars(num_buf.data(), num_buf.data() + num_buf.size(), result.score);
            local_string.append(num_buf.data(), ptr1 - num_buf.data());

            local_string += "/";

            auto [ptr2, ec2] = std::to_chars(num_buf.data(), num_buf.data() + num_buf.size(), DENOMINATORS[result.mask]);
            local_string.append(num_buf.data(), ptr2 - num_buf.data());

            local_string += "\n";
        }

        outfile << local_string;
        local_string.clear();

        total_written += elements_received;
        constexpr std::size_t TOTAL_TO_WRITE = num_canonical_hands;
        double frac = static_cast<double>(total_written) / static_cast<double>(TOTAL_TO_WRITE);
        double pct = frac * 100.0;
        auto now = std::chrono::steady_clock::now();
        auto time_taken = now - before;
        auto expected_total_time = time_taken / frac;
        std::string eta_string = std::format(
            "{:%M:%S}/{:%M:%S}",
            std::chrono::duration_cast<std::chrono::seconds>(time_taken),
            std::chrono::duration_cast<std::chrono::seconds>(expected_total_time));
        std::fprintf(stdout, "wrote %zu/%zu items (%.2f%%) [%s]\n", total_written, TOTAL_TO_WRITE, pct, eta_string.c_str());
    }
}

int main(int argc, char **argv) {
    using Hand = PokerHandJacksOrBetter;

    if (argc != 1) {
        std::cout << argv[0] << " takes no arguments.\n";
        return 1;
    }
    std::cout << "This is project " << PROJECT_NAME << ".\n";

    fprintf(stdout, "Filtering all hands to uniqueness by suit...\n");
    const std::vector<CanonicalHand> canonical_hands = generate_canonical_hands();
    fprintf(stdout, "Found %zu hands (expected %zu)\n", canonical_hands.size(), num_canonical_hands);

    std::ofstream outfile("results.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open results.txt\n";
        return 1;
    }
    std::thread writer(background_writer, std::move(outfile));


    #pragma omp parallel
    {
        moodycamel::ProducerToken write_production(write_queue);
        std::vector<HandResult> buffer;
        buffer.reserve(2048);

        #pragma omp for schedule(dynamic, 64)
        for (std::size_t i = 0; i < canonical_hands.size(); i++) {
            const auto& [hand, weight] = canonical_hands[i];

            const std::array<Card, 5> cards = mask_to_hand5(hand);
            const ScoreAfterDiscards scores = all_scores_for<Hand>(hand, cards);

            std::uint64_t best_mask = 0;
            std::int64_t best_score = std::numeric_limits<std::int64_t>::min();
            std::uint64_t best_denominator = 1;
            for (std::size_t discard_mask = 0; discard_mask <= 0b11111; discard_mask++) {
                const std::uint64_t current_denom = DENOMINATORS[discard_mask];
                const std::int64_t other_num = current_denom * best_score;
                const std::int64_t current_num = best_denominator * scores[discard_mask];

                if (current_num > other_num) {
                    best_mask = discard_mask;
                    best_score = scores[discard_mask];
                    best_denominator = current_denom;
                }
            }

            buffer.push_back(HandResult {
                .original_hand = hand,
                .mask = static_cast<uint8_t>(best_mask),
                .score = best_score,
                .weight = weight,
            });
            if (buffer.size() >= 2048) {
                write_queue.enqueue_bulk(write_production, buffer.begin(), buffer.size());
                buffer.clear();
            }
        }

        if (!buffer.empty()) {
            write_queue.enqueue_bulk(write_production, buffer.begin(), buffer.size());
            buffer.clear();
        }
    }

    computations_done.store(true, std::memory_order::release);

    writer.join();

    return 0;
}
