#include <particle.h>
#include <grid.h>
#include <cutil.h>

SphParticleSet2 *SphParticleSet2FromBuilder(ParticleSetBuilder2 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet2FromBuilder");
    ParticleSet2 *pSet = builder->MakeParticleSet();
    SphParticleSet2 *sphSet = cudaAllocateVx(SphParticleSet2, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

SphParticleSet2 *SphParticleSet2FromContinuousBuilder(ContinuousParticleSetBuilder2 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet2FromBuilder");
    ParticleSet2 *pSet = builder->GetParticleSet();
    SphParticleSet2 *sphSet = cudaAllocateVx(SphParticleSet2, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

SphParticleSet3 *SphParticleSet3FromBuilder(ParticleSetBuilder3 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet3FromBuilder");
    ParticleSet3 *pSet = builder->MakeParticleSet();
    SphParticleSet3 *sphSet = cudaAllocateVx(SphParticleSet3, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

SphParticleSet3 *SphParticleSet3FromContinuousBuilder(ContinuousParticleSetBuilder3 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet3FromBuilder");
    ParticleSet3 *pSet = builder->GetParticleSet();
    SphParticleSet3 *sphSet = cudaAllocateVx(SphParticleSet3, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

SphParticleSet2 *SphParticleSet2ExFromBuilder(ParticleSetBuilder2 *builder){
    AssertA(builder, "Invalid builder pointer for SphParticleSet2FromBuilder");
    ParticleSet2 *pSet = builder->MakeExtendedParticleSet();
    SphParticleSet2 *sphSet = cudaAllocateVx(SphParticleSet2, 1);
    sphSet->SetParticleData(pSet);
    return sphSet;
}

SpecieSet2 *SpecieSet2FromBuilder(ParticleSetBuilder2 *builder,
                                           Float mass, Float charge, int familyId)
{
    AssertA(builder, "Invalid builder pointer for SphParticleSet2FromBuilder");
    SpecieSet2 *pSet = builder->MakeSpecieSet(mass, charge);
    pSet->SetFamilyId(familyId);
    return pSet;
}
