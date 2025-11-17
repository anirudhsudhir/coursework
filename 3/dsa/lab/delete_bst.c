#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX1 25
#define MAX2 100

typedef struct node {
  char key[MAX1], value[MAX2];
  struct node *left, *right;
} NODE;

typedef struct tree {
  NODE *root;
} TREE;

// Function to initialize the tree
void init(TREE *pt);
// Function to create a new node
NODE *createNode(char word[MAX1], char meaning[MAX2]);
// Function to recursively insert a node into the BST
NODE *rinsert(NODE *r, NODE *temp);
// Function to insert a node into the tree
void recInsert(TREE *pt, char word[MAX1], char meaning[MAX2]);
// Function for inorder traversal of the BST
void inorder(NODE *r);
// Function to delete a node from the BST
NODE *delNode(NODE *r, char word[MAX1]);
// Function for iterative search in the BST
int search(NODE *r, char word[MAX1]);
// Function to destroy all nodes in the tree
void destroyNode(NODE *r);
// Function to destroy the tree
void destroyTree(TREE *pt);

int main() {
  TREE tobj;
  init(&tobj);
  int num_operations;
  char word[MAX1], meaning[MAX2];
  int choice;

  // printf("Enter the number of operations: ");
  scanf("%d", &num_operations);

  // Array to store search results (1 for success, 0 for failure)
  int search_results[num_operations];
  int search_index = 0;

  // Processing all operations
  for (int i = 0; i < num_operations; i++) {
    scanf("%d", &choice);

    switch (choice) {
    case 1: // Insert
      scanf("%s", word);
      fflush(stdin);

      scanf("%[^\n]", meaning);
      recInsert(&tobj, word, meaning);
      break;
    case 2: // Delete
      scanf("%s", word);
      tobj.root = delNode(tobj.root, word);
      break;
    case 3: // Search (results for search only)
      scanf("%s", word);
      if (search(tobj.root, word)) {
        search_results[search_index++] = 1; // Search success
      } else {
        search_results[search_index++] = 0; // Search failure
      }
      break;
    default:
      printf("opcode %d, ", choice);
      printf("Invalid operation\n");
    }
  }

  // Inorder traversal after all operations
  printf("Inorder Traversal:\n");
  inorder(tobj.root);

  // Print search results (1 for success, 0 for failure)
  printf("\nSearch Results:\n");
  for (int i = 0; i < search_index; i++) {
    printf("%d ", search_results[i]);
  }
  printf("\n");

  destroyTree(&tobj);
  return 0;
}

void init(TREE *pt) { pt->root = NULL; }

NODE *createNode(char word[MAX1], char meaning[MAX2]) {
  NODE *temp = malloc(sizeof(NODE));
  strcpy(temp->key, word);
  strcpy(temp->value, meaning);
  temp->left = NULL;
  temp->right = NULL;
  return temp;
}

// BST: Recursive Insert
NODE *rinsert(NODE *r, NODE *temp) {
  // complete the function for recursively inserting the words and their
  // meanings
  if (r == NULL) {
    return temp;
  } else if (strcmp(r->key, temp->key) > 0) {

    r->left = rinsert(r->left, temp);
  } else if (strcmp(r->key, temp->key) < 0) {
    r->right = rinsert(r->right, temp);
  }

  return r;
}

void recInsert(TREE *pt, char word[MAX1], char meaning[MAX2]) {
  NODE *temp = createNode(word, meaning);
  pt->root = rinsert(pt->root, temp);
}

void inorder(NODE *r) {

  // complete the function for inorder traversal
  if (r == NULL) {
    return;
  } else {
    inorder(r->left);
    printf("%s,%s\n", r->key, r->value);
    inorder(r->right);
  }
}

NODE *inorderSuccessor(NODE *r) {
  while (r != NULL && r->left != NULL) {
    r = r->left;
  }
  return r;
}

NODE *delNode(NODE *r, char word[MAX1]) {

  // complete function to delete the node and return the root

  NODE *temp = r;
  NODE *prev = temp;
  while (temp != NULL) {
    if (strcmp(temp->key, word) > 0) {
      prev = temp;
      temp = temp->left;
    } else if (strcmp(temp->key, word) < 0) {
      prev = temp;
      temp = temp->right;
    } else if (strcmp(temp->key, word) == 0) {

      // performing deletion
      if (temp->left == NULL) {
        NODE *temp1 = temp->right;
        free(temp);
        return NULL;
      } else if (temp->right == NULL) {
        NODE *temp1 = temp->left;
        free(temp);
        return temp1;
      } else {
        NODE *s = inorderSuccessor(temp->right);
        strcpy(temp->key, s->key);
        strcpy(temp->value, s->value);
        delNode(temp, s->key);
      }
    }
  }
  return r;
}

// BST: Iterative search
int search(NODE *r, char word[MAX1]) {
  // complete code to search for the given word
  NODE *temp = r;
  while (temp != NULL) {
    if (strcmp(temp->key, word) > 0) {
      temp = temp->left;
    } else if (strcmp(temp->key, word) < 0) {
      temp = temp->right;
    } else if (strcmp(temp->key, word) == 0) {
      return 1;
    }
  }

  return 0;
}

void destroyNode(NODE *r) {
  if (r != NULL) {
    destroyNode(r->left);
    destroyNode(r->right);
    free(r);
  }
}

void destroyTree(TREE *pt) {
  if (pt->root != NULL) {
    destroyNode(pt->root);
    pt->root = NULL;
  }
}
