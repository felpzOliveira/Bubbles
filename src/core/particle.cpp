#include <particle.h>
#include <cutil.h>

__host__ SphParticleSet2 *SphParticleSet2FromBuilder(ParticleSetBuilder2 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet2FromBuilder");
    ParticleSet2 *pSet = builder->MakeParticleSet();
    SphParticleSet2 *sphSet = cudaAllocateVx(SphParticleSet2, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

__host__ SphParticleSet3 *SphParticleSet3FromBuilder(ParticleSetBuilder3 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet3FromBuilder");
    ParticleSet3 *pSet = builder->MakeParticleSet();
    SphParticleSet3 *sphSet = cudaAllocateVx(SphParticleSet3, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

__host__ SphParticleSet2 *SphParticleSet2ExFromBuilder(ParticleSetBuilder2 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet2FromBuilder");
    ParticleSet2 *pSet = builder->MakeExtendedParticleSet();
    SphParticleSet2 *sphSet = cudaAllocateVx(SphParticleSet2, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

__host__ SpecieSet2 *SpecieSet2FromBuilder(ParticleSetBuilder2 *builder,
                                           Float mass, Float charge, int familyId)
{
    AssertA(builder, "Invalid builder pointer for SphParticleSet2FromBuilder");
    SpecieSet2 *pSet = builder->MakeSpecieSet(mass, charge);
    pSet->SetFamilyId(familyId);
    return pSet;
}
