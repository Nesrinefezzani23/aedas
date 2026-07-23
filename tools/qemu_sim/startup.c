/* =============================================================
 *  Startup bare-metal ARM Cortex-M7
 *  tools/qemu_sim/startup.c
 *
 *  Ce fichier est exécuté AVANT main().
 *  Il initialise la mémoire et configure le vecteur de reset.
 * ============================================================= */

#include <stdint.h>

/* Symboles définis par le linker script */
extern uint32_t _sdata, _edata;   /* Début/fin section .data en RAM */
extern uint32_t _sbss,  _ebss;    /* Début/fin section .bss  en RAM */
extern uint32_t _estack;           /* Sommet de la stack */

/* Déclaration de main */
extern int main(void);

/* Gestionnaires d'exceptions */
void Reset_Handler(void);
void Default_Handler(void) __attribute__((weak, alias("Reset_Handler")));
void HardFault_Handler(void) __attribute__((weak));

/* ── Table des vecteurs d'interruption ───────────────────────
 * Le Cortex-M7 lit cette table au démarrage à l'adresse 0x0.
 * Le premier mot = adresse initiale du stack pointer.
 * Le second mot  = adresse du Reset_Handler.
 */
__attribute__((section(".isr_vector")))
const uint32_t vector_table[] = {
    (uint32_t)&_estack,            /* Stack Pointer initial */
    (uint32_t)Reset_Handler,       /* Reset */
    (uint32_t)Default_Handler,     /* NMI */
    (uint32_t)HardFault_Handler,   /* HardFault */
    (uint32_t)Default_Handler,     /* MemManage */
    (uint32_t)Default_Handler,     /* BusFault */
    (uint32_t)Default_Handler,     /* UsageFault */
    0, 0, 0, 0,                    /* Réservés */
    (uint32_t)Default_Handler,     /* SVCall */
    (uint32_t)Default_Handler,     /* DebugMonitor */
    0,                             /* Réservé */
    (uint32_t)Default_Handler,     /* PendSV */
    (uint32_t)Default_Handler,     /* SysTick */
};

/* ── Reset Handler ───────────────────────────────────────────
 * Première fonction exécutée après le reset du CPU.
 * 1. Copie .data de Flash vers RAM
 * 2. Met .bss à zéro
 * 3. Appelle main()
 */
void Reset_Handler(void) {
    /* Copie .data (Flash → RAM) */
    uint32_t *src = (uint32_t*)((uint32_t)&_sdata +
                    ((uint32_t)&_edata - (uint32_t)&_sdata));
    uint32_t *dst = &_sdata;

    /* En réalité : src pointe après .text en Flash */
    /* Simplifié pour QEMU qui initialise déjà la RAM */

    /* Mise à zéro de .bss */
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    /* Appel de main */
    main();

    /* Si main() retourne (ne devrait pas) : boucle infinie */
    while(1);
}