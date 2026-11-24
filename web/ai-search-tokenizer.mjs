const BYTE_OFFSET = 4;
const BASE_SIZE = BYTE_OFFSET + 256;
const SEGMENTS = /\\[A-Za-z]+|[\p{L}\p{Nl}\p{No}]+|\p{Nd}+|[ \t]+|\r\n|[^\p{L}\p{N}_\p{White_Space}\u001c-\u001f]+|[\s\S]/gu;
const encoder = new TextEncoder();

export class UniversityBPETokenizer {
  constructor(json) {
    if (json.version !== 1 || json.algorithm !== "byte-bpe" || json.special?.length !== 4) {
      throw new Error("Unsupported University tokenizer format");
    }
    this.merges = json.merges;
    this.vocabSize = json.vocab_size;
    this.ranks = new Map();
    this.merges.forEach((pair, i) => this.ranks.set(`${pair[0]}:${pair[1]}`, BASE_SIZE + i));
  }

  encodeSegment(segment) {
    let ids = Array.from(encoder.encode(segment), b => b + BYTE_OFFSET);
    while (ids.length > 1) {
      let bestRank = this.vocabSize;
      let bestPair = null;
      for (let i = 0; i + 1 < ids.length; i++) {
        const key = `${ids[i]}:${ids[i + 1]}`;
        const rank = this.ranks.get(key) ?? this.vocabSize;
        if (rank < bestRank) { bestRank = rank; bestPair = key; }
      }
      if (bestPair === null) break;
      const merged = bestRank;
      const [left, right] = bestPair.split(":").map(Number);
      const next = [];
      for (let i = 0; i < ids.length;) {
        if (i + 1 < ids.length && ids[i] === left && ids[i + 1] === right) {
          next.push(merged); i += 2;
        } else next.push(ids[i++]);
      }
      ids = next;
    }
    return ids;
  }

  encode(text) {
    const pieces = text.match(SEGMENTS) ?? [];
    const body = pieces.flatMap(piece => this.encodeSegment(piece));
    return [1, ...body, 2];
  }

  inputs(text, maxLength = 512) {
    const ids = this.encode(text);
    if (ids.length > maxLength) throw new Error(`Query has ${ids.length} tokens; limit is ${maxLength}`);
    return { input_ids: ids, attention_mask: ids.map(() => 1) };
  }
}
