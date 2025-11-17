#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int val;
  bool set;
} Value;

typedef struct {
  int TableSize;
  int NumElements;
  Value *Table;
} Hashtable;

Hashtable *initHashtable(int);
bool insert(Hashtable *, int, int);
void displayHashtable(Hashtable *);

int hash1(int, int);
int hash2(int, int);

int main() {
  int hashtableSize, numEle;
  scanf("%d", &hashtableSize);
  scanf("%d", &numEle);

  Hashtable *store = initHashtable(hashtableSize);

  for (int i = 0; i < numEle; i++) {
    int val;
    scanf("%d", &val);
    insert(store, i, val);
  }

  displayHashtable(store);

  return 0;
}

bool insert(Hashtable *hashtable, int key, int val) {
  int primaryHash = hash1(val, hashtable->TableSize);
  if (!hashtable->Table[primaryHash].set) {
    hashtable->Table[primaryHash].val = val;
    hashtable->Table[primaryHash].set = true;
    hashtable->NumElements++;
    // printf("\ninserted key %d, val %d at index %d with primary hash %d\n",
    // key,
    //        val, primaryHash, primaryHash);

    return true;
  }

  int secondaryHash = hash2(val, hashtable->TableSize);
  int index;
  for (int i = 1; i < hashtable->TableSize; i++) {
    index = (primaryHash + i * secondaryHash) % hashtable->TableSize;
    if (!hashtable->Table[index].set) {
      hashtable->Table[index].val = val;
      hashtable->Table[index].set = true;
      hashtable->NumElements++;
      // printf("\ninserted key %d, val %d at index %d with double hash %d\n",
      // key,
      //        val, index, secondaryHash);

      return true;
    }
  }
  return false;
}

int hash1(int key, int hashtableSize) { return (2 * key + 3) % hashtableSize; }
int hash2(int key, int hashtableSize) { return (3 * key + 1) % hashtableSize; }

Hashtable *initHashtable(int hashtableSize) {
  Hashtable *store = (Hashtable *)malloc(sizeof(Hashtable));
  store->NumElements = 0;
  store->TableSize = hashtableSize;
  store->Table = (Value *)malloc(hashtableSize * sizeof(Value));

  for (int i = 0; i < hashtableSize; i++) {
    store->Table[i].set = false;
    store->Table[i].val = 0;
  }

  return store;
}

void displayHashtable(Hashtable *hashtable) {
  for (int i = 0; i < hashtable->TableSize; i++) {
    if (hashtable->Table[i].set) {
      printf("%d\n", hashtable->Table[i].val);
    } else {
      printf("-1\n");
    }
  }
}
