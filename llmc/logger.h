/*
Implements a simple logger that writes log files in the output directory.
The Logger object is stateless and uses append mode to write to log files.
*/
#ifndef LOGGER_H
#define LOGGER_H

#include <assert.h>
#include <stdio.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

typedef struct {
    int active;
    char output_log_file[512];
    char output_log_dir[512];
    int resume;
} Logger;

void logger_init(Logger *logger, const char *log_dir, int process_rank, int resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    logger->output_log_dir[0] = '\0';
    logger->resume = resume;
    if (log_dir != NULL && process_rank == 0) {
        logger->active = 1;
        assert(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_dir, 512, "%s", log_dir);
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        if (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

void logger_log_eval(Logger *logger, int step, float val) {
    if (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

void logger_log_val_metrics_txt(Logger *logger, const char *model_name, int step,
                                float val_loss, float val_perplexity) {
    if (logger->active == 1) {
        char metrics_file[512];
        if (model_name == NULL || model_name[0] == '\0' ||
            logger->output_log_dir[0] == '\0') {
            return;
        }

        size_t dir_len =
            strnlen(logger->output_log_dir, sizeof(logger->output_log_dir));
        size_t model_len = strnlen(model_name, 256);
        const size_t suffix_len = sizeof(".txt") - 1;

        // Need: <dir>/<model>.txt\0
        if (dir_len + 1 + model_len + suffix_len + 1 > sizeof(metrics_file)) {
            return;
        }

        memcpy(metrics_file, logger->output_log_dir, dir_len);
        metrics_file[dir_len] = '/';
        memcpy(metrics_file + dir_len + 1, model_name, model_len);
        memcpy(metrics_file + dir_len + 1 + model_len, ".txt",
               suffix_len + 1); // also copy null terminator

        const char *mode = (step == 0 && logger->resume == 0) ? "w" : "a";
        FILE *logfile = fopenCheck(metrics_file, mode);
        fprintf(logfile, "step:%d val_loss:%.6f val_perplexity:%.6f\n", step,
                val_loss, val_perplexity);
        fclose(logfile);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss, float learning_rate, float grad_norm) {
    if (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif