#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct node {
  char name[50];
  char text[50];
  struct node *prev;
  struct node *next;
} chat;

void disp(chat *headp, chat *head) {
  chat *hp = headp;
  if (head == NULL) {
    printf("Empty\n");
  } else {
    int a = 0;
    if (headp != NULL) {
      printf("Pinned Chats\n");
      while (headp != head->next && headp != NULL) {
        printf("name:%s\n", headp->name);
        printf("text:%s\n", headp->text);
        headp = headp->next;
        if (a++ == 10) {
          break;
        }
      }
    }

    printf("Unpinned Chats\n");
    if (hp != NULL) {
      head = head->next;
    }
    int b = 0;
    while (head != NULL) {
      printf("name:%s\n", head->name);
      printf("text:%s\n", head->text);
      head = head->next;

      if (b++ == 10) {
        printf("\nUNPINNED BREAK\n");
        break;
      }
    }
  }
}

chat *addchat(chat *headp, chat *head, char *n, char *t) {
  chat *new_chat = (chat *)malloc(sizeof(chat));
  new_chat->prev = NULL;
  new_chat->next = NULL;
  strcpy(new_chat->name, n);
  strcpy(new_chat->text, t);
  if (head == NULL) {
    return new_chat;
  }
  if (headp == NULL) {
    // No pinned chats, head points to first unpinned chat
    new_chat->next = head;
    head->prev = new_chat;
    return new_chat;
  } else {
    // If no unpinned chats remain
    if (head->next == NULL) {
      head->next = new_chat;
      new_chat->prev = head;
    }
    // If unpinned chats remain
    else {
      chat *first_unpinned_chat = head->next;
      new_chat->next = head->next;
      head->next = new_chat;
      new_chat->prev = head;
      new_chat->next->prev = new_chat;
    }

    return head;
  }
}

void pinchat(chat **headp, chat **head, chat *p) {
  // no unpinned chats
  if (*head == NULL) {
    return;
  }

  chat *chat_to_pin = *head;
  if (*headp != NULL) {
    chat_to_pin = (*head)->next;
  }

  while (strcmp(chat_to_pin->name, p->name) != 0) {
    chat_to_pin = chat_to_pin->next;
  }

  chat *prev_chat = chat_to_pin->prev;
  chat *next_chat = chat_to_pin->next;

  if (prev_chat != NULL && next_chat != NULL) {
    prev_chat->next = next_chat;
    next_chat->prev = prev_chat;
  } else if (prev_chat != NULL) {
    prev_chat->next = NULL;
  } else if (next_chat != NULL) {
    next_chat->prev = NULL;
  }

  if (headp == NULL && (*head)->next == NULL) {
    *head = NULL;
  }

  // no pinned chats yet
  if (*headp == NULL) {
    chat_to_pin->next = *head;
    chat_to_pin->prev = NULL;
    *headp = chat_to_pin;
    *head = chat_to_pin;
    // unpinned chats exist
    if (*head != NULL) {
      chat_to_pin->next->prev = chat_to_pin;
    }
  }
  // pinned chats exist, add chat to end of pinned chats
  else {
    chat_to_pin->next = (*head)->next;
    (*head)->next = chat_to_pin;
    chat_to_pin->prev = (*head);
    *head = chat_to_pin;
    // unpinned chats exist
    if ((*head)->next != NULL) {
      (*head)->next->prev = chat_to_pin;
    }
  }
}

void unpin(chat **headp, chat **head, chat *p) {
  // no pinned chats
  if (*headp == NULL) {
    return;
  }

  chat *chat_to_unpin = *headp;
  while (strcmp(chat_to_unpin->name, p->name) != 0) {
    chat_to_unpin = chat_to_unpin->next;
  }

  chat *prev_chat = chat_to_unpin->prev;
  chat *next_chat = chat_to_unpin->next;

  if (prev_chat != NULL && next_chat != NULL) {
    prev_chat->next = next_chat;
    next_chat->prev = prev_chat;
  } else if (prev_chat != NULL) {
    prev_chat->next = NULL;
  } else if (next_chat != NULL) {
    next_chat->prev = NULL;
  }

  // deal with headp and head
  if ((*headp) == (*head)) {
    *headp = NULL;
    *head = (*head)->next;
  } else {
    if ((*headp) == chat_to_unpin) {
      *headp = next_chat;
    } else if ((*head) == chat_to_unpin) {
      *head = prev_chat;
    }
  }

  // no pinned or unpinned chats
  if (*head == NULL) {
    (*head) = chat_to_unpin;
    chat_to_unpin->next = NULL;
    chat_to_unpin->prev = NULL;
  }
  // unpinned chats exist
  else {
    // head pointing to first unpinned chats
    if (*headp == NULL) {
      chat_to_unpin->next = *head;
      (*head)->prev = chat_to_unpin;
      chat_to_unpin->prev = NULL;
      *head = chat_to_unpin;
    }
    // head pointing to final pinned chat
    else {
      chat_to_unpin->next = (*head)->next;
      chat_to_unpin->prev = (*head);
      if ((*head)->next != NULL) {
        (*head)->next->prev = chat_to_unpin;
      }
      (*head)->next = chat_to_unpin;
    }
  }
}

int main() {
  chat *head = NULL;  // pointer to first node before unpinned chats
  chat *headp = NULL; // pointer to first node amongst pinned chats
  int out = 0;
  printf("***************************************************************\n");
  printf("YOUR CHAT MESSENGER\n");
  printf("***************************************************************\n");

  while (out == 0) {
    printf("Pick your action:\n");
    printf("To DISPLAY all chats enter 0\n");
    printf("To ADD a chat enter 1\n");
    printf("To PIN a chat print 2\n");
    printf("To UNPIN a chat enter 3\n");
    int op = 0; // takes an option
    printf("Enter option:");
    scanf("%d", &op);
    if (op == 1) {
      char n[50] = ""; // stores name of chat
      char t[50] = ""; // stores text in chat
      printf("Enter name:");
      getchar();
      scanf("%[^\n]s", n);
      getchar();
      printf("Enter text:");
      scanf("%[^\n]s", t);
      head = addchat(headp, head, n, t);
    } else if (op == 2) {
      printf("Enter name of contact from unpinned chats:\n");
      getchar();
      char cont[50] = ""; // name of contact
      disp(headp, head);
      scanf("%[^\n]s", cont);
      chat *p = head;
      while (strcmp(p->name, cont) !=
             0) // moves the pointer to the specific chat
      {
        p = p->next;
      }

      pinchat(&headp, &head, p);
    }

    else if (op == 3) {
      printf("From the pinned chats enter a chat that you want to unpin\n");
      disp(headp, head);
      char in[50] = "";
      getchar();
      scanf("%[^\n]s", in);
      chat *point2 = headp;
      while (strcmp(point2->name, in) != 0) {
        point2 = point2->next;
      }
      unpin(&headp, &head, point2);
    } else if (op == 0) {
      disp(headp, head);
    } else {
      printf("Wrong Input\n");
    }
  }
}
