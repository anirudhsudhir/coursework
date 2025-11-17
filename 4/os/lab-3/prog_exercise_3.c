#include <stdio.h>
#include <stdlib.h>

#define CACHE_SIZE 3
#define PAGE_TABLE_SIZE 1024

typedef struct page {
  int page_id;
} page;

typedef struct frame {
  struct frame *prev;
  struct frame *next;
  struct page *pg;
} frame;

typedef struct {
  // Cache holding three frames
  frame *cache;
  int current_cache_size;

  // Array holding all pages
  frame *frames[PAGE_TABLE_SIZE];

  int pg_fault_count;
} page_table;

void run_loop(page_table *pg_tbl);
page *new_page(int data);

void insert_page(page_table *pg_tbl, page *pg);
page *access_page(page_table *pg_tbl, int pg_id);

int main(void) {
  page_table *pg_tbl = (page_table *)malloc(sizeof(page_table));
  pg_tbl->cache = NULL;
  pg_tbl->current_cache_size = 0;
  pg_tbl->pg_fault_count = 0;
  for (int i = 0; i < PAGE_TABLE_SIZE; i++) {
    pg_tbl->frames[i] = NULL;
  }
  // for (int i = 0; i < CACHE_SIZE; i++) {
  //   pg_tbl->cache[i] = NULL;
  // }

  run_loop(pg_tbl);
  return 0;
}

void run_loop(page_table *pg_tbl) {
  int choice;
  while (1) {
    printf("\nChoose an operation:\n");
    printf("1. Insert a new page\n");
    printf("2. Retrieve a page\n");
    // printf("3. Remove a page\n");
    printf("3. Exit\n");
    scanf("%d", &choice);

    switch (choice) {
    case 1:
      printf("Enter the id of the page as an integer\n");
      int data;
      scanf("%d", &data);
      page *pg = new_page(data);
      insert_page(pg_tbl, pg);
      break;

    case 2:
      printf("Enter the id of the page to be accessed\n");
      int pg_id;
      scanf("%d", &pg_id);
      page *search_pg = access_page(pg_tbl, pg_id);
      if (search_pg == NULL) {
        printf("Page not found!\n");
      } else {
        printf("Page id = %d\n", search_pg->page_id);
      }

      printf("Page Table\n");
      for (int i = 0; i < PAGE_TABLE_SIZE; i++) {
        if (pg_tbl->frames[i] != NULL)
          printf("%d ", pg_tbl->frames[i]->pg->page_id);
      }
      printf("\n");

      printf("Cache: ");
      frame *head = pg_tbl->cache;
      while (head != NULL) {
        printf("%d ", head->pg->page_id);
        head = head->next;
      }
      printf("\n");
      break;

    case 3:
      return;

    default:
      printf("Invalid choice!\n");
    }
  }
}

page *new_page(int pg_id) {
  page *pg = (page *)malloc(sizeof(page));
  pg->page_id = pg_id;
  return pg;
}

void insert_page(page_table *pg_tbl, page *pg) {
  frame *fm = (frame *)malloc(sizeof(frame));
  fm->next = NULL;
  fm->prev = NULL;
  fm->pg = pg;
  pg_tbl->frames[pg->page_id] = fm;
  printf("Inserted page %d\n", pg->page_id);

  printf("Page Table: ");
  for (int i = 0; i < PAGE_TABLE_SIZE; i++) {
    if (pg_tbl->frames[i] != NULL)
      printf("%d ", pg_tbl->frames[i]->pg->page_id);
  }
  printf("\n");

  printf("Cache: ");
  frame *head = pg_tbl->cache;
  while (head != NULL) {
    printf("%d ", head->pg->page_id);
  }
  printf("\n");
}

page *access_page(page_table *pg_tbl, int pg_id) {

  frame *head = pg_tbl->cache;
  while (head != NULL) {
    if (head->pg->page_id == pg_id) {
      printf("Cache hit!\n");

      if (head->prev != NULL) {
        head->prev->next = head->next;
        if (head->next != NULL) {
          head->next->prev = head->prev;
        }
        head->next = pg_tbl->cache;
        pg_tbl->cache->prev = head;
        pg_tbl->cache = head;
      }
      return head->pg;
    }
    head = head->next;
  }

  // cache miss, page fault
  frame *fm = pg_tbl->frames[pg_id];
  if (fm == NULL) {
    return NULL;
  }

  pg_tbl->pg_fault_count++;
  printf("Cache miss!\n");
  printf("Page faults = %d\n", pg_tbl->pg_fault_count);

  // lru cache update
  if (pg_tbl->cache == NULL) {
    // cache is empty
    pg_tbl->cache = fm;
    pg_tbl->current_cache_size++;
  } else {

    if (pg_tbl->current_cache_size < CACHE_SIZE) {
      fm->next = pg_tbl->cache;
      pg_tbl->cache->prev = fm;
      pg_tbl->cache = fm;
      pg_tbl->current_cache_size++;
    } else {
      // perform lru cache eviction

      printf("Evicting frame\n");
      frame *temp = pg_tbl->cache;
      while (temp->next != NULL) {
        temp = temp->next;
      }

      temp->prev->next = NULL;
      printf("Victim page = %d\n", temp->pg->page_id);

      fm->next = pg_tbl->cache;
      pg_tbl->cache->prev = fm;
      pg_tbl->cache = fm;
    }
  }

  return fm->pg;
}
