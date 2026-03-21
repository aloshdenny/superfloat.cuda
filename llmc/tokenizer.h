/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#include <assert.h>
#include <ctype.h>
#include <stdint.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------

typedef struct {
  uint32_t vocab_size;
  char **token_table;
  int init_ok;
  int eot_token; // <|endoftext|> token id
  int contiguous_payload; // 1 when token strings share one allocation
} Tokenizer;

enum {
  TOKENIZER_MAGIC_STD = 20240328,
  TOKENIZER_MAGIC_Q15 = 2024,
  TOKENIZER_MAGIC_SF16 = 20260321
};

void safe_printf(const char *piece) {
  // the tokens are raw bytes, and we we only want to print the printable ones
  // many bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  // handle individual byte tokens
  // every token is asserted to be at least one byte so doing piece[1] is ok
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // weird byte, don't print it
    }
  }
  printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    // try to be more helpful as we just added this feature, erase later
    printf("---\n");
    printf("WARNING: Failed to open the tokenizer file %s\n", filename);
    printf("The Tokenizer is a new feature added April 14 2024.\n");
    printf("Re-run `python train_gpt2.py` to write it\n");
    printf("---\n");
    tokenizer->init_ok = 0;
    tokenizer->contiguous_payload = 0;
    return;
  }
  // read in the header
  uint32_t header[256];
  freadCheck(header, sizeof(uint32_t), 256, file);
  assert(header[0] == TOKENIZER_MAGIC_STD);
  int version = header[1];
  tokenizer->vocab_size = header[2];
  if (version == 1) {
    // version 1 didn't include the EOT token id
    // so we assume it is 50256, the EOT in GPT-2
    assert(tokenizer->vocab_size == 50257); // let's be defensive here
    tokenizer->eot_token = 50256;
  } else if (version == 2) {
    tokenizer->eot_token = header[3];
  } else {
    fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename,
            version);
    exit(EXIT_FAILURE);
  }
  // read in all the tokens
  unsigned char length;
  tokenizer->token_table =
      (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
  for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
    freadCheck(&length, sizeof(unsigned char), 1, file);
    assert(length > 0); // every token should be at least one character
    char *token_bytes = (char *)mallocCheck(length + 1);
    freadCheck(token_bytes, sizeof(char), length, file);
    token_bytes[length] = '\0'; // Add null terminator for printing
    tokenizer->token_table[i] = token_bytes;
  }
  // cleanups
  fcloseCheck(file);
  tokenizer->init_ok = 1;
  tokenizer->contiguous_payload = 0;
}

// SF16 tokenizer binary format:
// header[0] = TOKENIZER_MAGIC_SF16
// header[1] = version (1)
// header[2] = vocab_size (uint32)
// header[3] = eot_token (uint32)
// header[4] = payload_bytes (uint32) including all token bytes plus nulls
// data section:
//   uint32 lengths[vocab_size]
//   char payload[payload_bytes] as NUL-terminated token strings packed back-to-back
void tokenizer_init_sf16(Tokenizer *tokenizer, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    fprintf(stderr, "Failed to open SF16 tokenizer file: %s\n", filename);
    tokenizer->init_ok = 0;
    tokenizer->contiguous_payload = 0;
    return;
  }

  uint32_t header[256];
  freadCheck(header, sizeof(uint32_t), 256, file);
  if (header[0] != TOKENIZER_MAGIC_SF16) {
    fprintf(stderr, "SF16 tokenizer %s has bad magic: %u\n", filename, header[0]);
    exit(EXIT_FAILURE);
  }
  if (header[1] != 1u) {
    fprintf(stderr, "SF16 tokenizer %s has bad version: %u\n", filename, header[1]);
    exit(EXIT_FAILURE);
  }

  tokenizer->vocab_size = header[2];
  tokenizer->eot_token = (int)header[3];
  uint32_t payload_bytes = header[4];
  if (tokenizer->vocab_size == 0 || payload_bytes == 0) {
    fprintf(stderr, "SF16 tokenizer %s has invalid header sizes\n", filename);
    exit(EXIT_FAILURE);
  }

  tokenizer->token_table =
      (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
  uint32_t *lengths =
      (uint32_t *)mallocCheck(tokenizer->vocab_size * sizeof(uint32_t));
  freadCheck(lengths, sizeof(uint32_t), tokenizer->vocab_size, file);

  char *payload = (char *)mallocCheck(payload_bytes);
  freadCheck(payload, sizeof(char), payload_bytes, file);

  uint64_t cursor = 0;
  for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
    uint32_t len = lengths[i];
    if (len == 0 || cursor + (uint64_t)len + 1 > payload_bytes) {
      fprintf(stderr, "SF16 tokenizer %s has invalid token payload at id=%u\n",
              filename, i);
      exit(EXIT_FAILURE);
    }
    tokenizer->token_table[i] = payload + cursor;
    if (payload[cursor + len] != '\0') {
      fprintf(stderr, "SF16 tokenizer %s token %u is missing terminator\n",
              filename, i);
      exit(EXIT_FAILURE);
    }
    cursor += (uint64_t)len + 1;
  }

  free(lengths);
  fcloseCheck(file);
  tokenizer->init_ok = 1;
  tokenizer->contiguous_payload = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
  if (tokenizer->init_ok == 0) {
    return NULL;
  }
  if (token_id < tokenizer->vocab_size) {
    return tokenizer->token_table[token_id];
  } else {
    printf("invalid token id %u!\n", token_id);
    return NULL;
  }
}

void tokenizer_free(Tokenizer *tokenizer) {
  if (tokenizer->init_ok) {
    if (tokenizer->contiguous_payload) {
      free(tokenizer->token_table[0]);
    } else {
      for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->token_table[i]);
      }
    }
    free(tokenizer->token_table);
  }
}

void tokenizer_init_q15(Tokenizer *tokenizer, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    // If the Q1.15 file is missing, try to reconstruct it from the standard one
    const char *fallback_filename = "gpt2_tokenizer.bin";
    FILE *fallback_file = fopen(fallback_filename, "rb");
    if (fallback_file == NULL) {
      printf("---\n");
      printf("WARNING: Failed to open both the q1.15 tokenizer file '%s' and "
             "the fallback '%s'\n",
             filename, fallback_filename);
      printf("---\n");
      tokenizer->init_ok = 0;
      tokenizer->contiguous_payload = 0;
      return;
    }

    // Read the standard header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, fallback_file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
      assert(tokenizer->vocab_size == 50257);
      tokenizer->eot_token = 50256;
    } else if (version == 2) {
      tokenizer->eot_token = header[3];
    } else {
      fprintf(stderr, "Tokenizer fallback model file %s has bad version: %d\n",
              fallback_filename, version);
      exit(EXIT_FAILURE);
    }

    // Read in all the tokens from the fallback
    unsigned char length;
    tokenizer->token_table =
        (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));

    printf("Reconstructing %s from %s...\n", filename, fallback_filename);
    FILE *out_file = fopen(filename, "wb");
    if (out_file != NULL) {
      uint16_t header_q15[256] = {0};
      header_q15[0] = 2024;
      header_q15[1] = 2;
      header_q15[2] = (uint16_t)tokenizer->vocab_size;
      header_q15[3] = (uint16_t)tokenizer->eot_token;
      fwrite(header_q15, sizeof(uint16_t), 256, out_file);
    }

    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
      freadCheck(&length, sizeof(unsigned char), 1, fallback_file);
      assert(length > 0);
      char *token_bytes = (char *)mallocCheck(length + 1);
      freadCheck(token_bytes, sizeof(char), length, fallback_file);
      token_bytes[length] = '\0';
      tokenizer->token_table[i] = token_bytes;

      if (out_file != NULL) {
        uint16_t len_q15 = (uint16_t)length;
        fwrite(&len_q15, sizeof(uint16_t), 1, out_file);
        for (int j = 0; j < (int)length; j++) {
          uint16_t c = (uint16_t)((unsigned char)token_bytes[j]);
          fwrite(&c, sizeof(uint16_t), 1, out_file);
        }
      }
    }
    fcloseCheck(fallback_file);
    if (out_file != NULL) {
      fclose(out_file);
      printf("Successfully constructed %s\n", filename);
    }
    tokenizer->init_ok = 1;
    tokenizer->contiguous_payload = 0;
    return;
  }
  // read in the header
  uint16_t header[256];
  freadCheck(header, sizeof(uint16_t), 256, file);
  assert(header[0] == 2024);
  int version = header[1];
  tokenizer->vocab_size = header[2];

  if (version == 1) {
    assert(tokenizer->vocab_size == 50257);
    tokenizer->eot_token = 50256;
  } else if (version == 2) {
    tokenizer->eot_token = header[3];
  } else {
    fprintf(stderr, "Tokenizer q1.15 model file %s has bad version: %d\n",
            filename, version);
    exit(EXIT_FAILURE);
  }

  // read in all the tokens
  uint16_t length;
  tokenizer->token_table =
      (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
  for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
    freadCheck(&length, sizeof(uint16_t), 1, file);
    assert(length > 0); // every token should be at least one character
    char *token_bytes = (char *)mallocCheck(length + 1);
    for (int j = 0; j < (int)length; j++) {
      uint16_t c;
      freadCheck(&c, sizeof(uint16_t), 1, file);
      token_bytes[j] = (char)c;
    }
    token_bytes[length] = '\0'; // Add null terminator for printing
    tokenizer->token_table[i] = token_bytes;
  }
  // cleanups
  fcloseCheck(file);
  tokenizer->init_ok = 1;
  tokenizer->contiguous_payload = 0;
}

const char *tokenizer_decode_q15(Tokenizer *tokenizer, uint16_t q1_15_id) {
  if (tokenizer->init_ok == 0) {
    return NULL;
  }
  // As per method A: ID is stored directly, so just cast to uint32_t
  uint32_t token_id = (uint32_t)q1_15_id;
  if (token_id < tokenizer->vocab_size) {
    return tokenizer->token_table[token_id];
  } else {
    printf("invalid token id %u!\n", token_id);
    return NULL;
  }
}
