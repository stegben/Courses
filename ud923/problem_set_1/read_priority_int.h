#ifndef READ_PRIORITY_INT
#define READ_PRIORITY_INT
#include <pthread.h>
#include <stdlib.h>

typedef struct ReadPriorityInt {
  int val;
  pthread_mutex_t val_lock;
  pthread_cond_t read_cond;
  pthread_cond_t write_cond;
} ReadPriorityInt;


ReadPriorityInt *init_read_priority_int(int value) {
  ReadPriorityInt* data_ptr = malloc(sizeof(ReadPriorityInt));
  data_ptr->val = value;
  pthread_mutex_init(&data_ptr->val_lock, NULL);
  pthread_cond_init(&data_ptr->read_cond, NULL);
  pthread_cond_init(&data_ptr->write_cond, NULL);
  return data_ptr;
}


void* read(ReadPriorityInt* data, int wait_sec);
void* write(ReadPriorityInt* data, int* val, int wait_sec);


#endif  // READ_PRIORITY_INT
