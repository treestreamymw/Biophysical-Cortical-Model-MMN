#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _ar_reg(void);
extern void _cad_reg(void);
extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _ca_reg(void);
extern void _cat_reg(void);
extern void _epsp_reg(void);
extern void _fdsexp2syn_reg(void);
extern void _Gfluct_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _kca_reg(void);
extern void _km_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _kv_reg(void);
extern void _na_2_reg(void);
extern void _Nap_Et2_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTs2_t_reg(void);
extern void _NMDA_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);
extern void _vecevent_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," mod_files//ar.mod");
    fprintf(stderr," mod_files//cad.mod");
    fprintf(stderr," mod_files//CaDynamics_E2.mod");
    fprintf(stderr," mod_files//Ca_HVA.mod");
    fprintf(stderr," mod_files//Ca_LVAst.mod");
    fprintf(stderr," mod_files//ca.mod");
    fprintf(stderr," mod_files//cat.mod");
    fprintf(stderr," mod_files//epsp.mod");
    fprintf(stderr," mod_files//fdsexp2syn.mod");
    fprintf(stderr," mod_files//Gfluct.mod");
    fprintf(stderr," mod_files//Ih.mod");
    fprintf(stderr," mod_files//Im.mod");
    fprintf(stderr," mod_files//kca.mod");
    fprintf(stderr," mod_files//km.mod");
    fprintf(stderr," mod_files//K_Pst.mod");
    fprintf(stderr," mod_files//K_Tst.mod");
    fprintf(stderr," mod_files//kv.mod");
    fprintf(stderr," mod_files//na_2.mod");
    fprintf(stderr," mod_files//Nap_Et2.mod");
    fprintf(stderr," mod_files//NaTa_t.mod");
    fprintf(stderr," mod_files//NaTs2_t.mod");
    fprintf(stderr," mod_files//NMDA.mod");
    fprintf(stderr," mod_files//SK_E2.mod");
    fprintf(stderr," mod_files//SKv3_1.mod");
    fprintf(stderr," mod_files//vecevent.mod");
    fprintf(stderr, "\n");
  }
  _ar_reg();
  _cad_reg();
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _ca_reg();
  _cat_reg();
  _epsp_reg();
  _fdsexp2syn_reg();
  _Gfluct_reg();
  _Ih_reg();
  _Im_reg();
  _kca_reg();
  _km_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _kv_reg();
  _na_2_reg();
  _Nap_Et2_reg();
  _NaTa_t_reg();
  _NaTs2_t_reg();
  _NMDA_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
  _vecevent_reg();
}
