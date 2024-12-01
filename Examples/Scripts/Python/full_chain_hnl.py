#!/usr/bin/env python3
import argparse, pathlib, contextlib, acts, acts.examples
from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    ParticleConfig,
    addPythia8,
    addFatras,
    addGeant4,
    ParticleSelectorConfig,
    addDigitization
)
from acts.examples.reconstruction import addSpacePointsMaking

from common import getOpenDataDetectorDirectory
from acts.examples.odd import getOpenDataDetector

parser = argparse.ArgumentParser(description="Full chain with the OpenDataDetector")

parser.add_argument("--events", "-n", help="Number of events", type=int, default=5)

args = vars(parser.parse_args())

u = acts.UnitConstants
geoDir = getOpenDataDetectorDirectory()
outputDir = pathlib.Path.cwd() / "HNL_output"

oddMaterialMap = geoDir / "data/odd-material-maps.root"
#oddFieldMap = geoDir / "data/odd-field.root"
oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
# oddDigiConfig = geoDir / "config/odd-digi-geometric-config.json"
oddSpacepointSel = geoDir / "config/odd-sp-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(
    geoDir, mdecorator=oddMaterialDeco
)
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=411)

ttbar_pu200=False

# TODO Geant4 currently crashes with FPE monitoring
with acts.FpeMonitor():
    s = acts.examples.Sequencer(
        events=args["events"],
        numThreads=8,
        outputDir=str(outputDir))

    addPythia8(
        s,
        hardProcess=["50:new = N2 N2 2 0 0 15.0 0.0 0.0 0.0 100.0 0 1 0 1 0 ",
                     "50:isResonance = false",
                     "50:addChannel = 1 0.50 23 -13  13 -14",
                     "50:addChannel = 1 0.50 23  13 -13 14",
                     "50:mayDecay = on",
                     "WeakSingleBoson:ffbar2W = on",
                     "24:onMode = off",
                     "24:addchannel = 1 1.0 103 -13 50",
                     "ParticleDecays:limitTau0 = off",
                     "ParticleDecays:tau0Max = 600.0",
                     "Tune:ee = 7",
                     "Tune:pp = 21"
                     ],
        npileup=0,
        vtxGen=acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(
                0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns
            ),
            mean=acts.Vector4(0, 0, 0, 0),
        ),
        rnd=rnd,
        outputDirCsv=outputDir,
    )
  
    addFatras(
        s,
        trackingGeometry,
        field,
        preSelectParticles=ParticleSelectorConfig(
#            eta=(-3.0, 3.0),
            pt=(1.0 * u.GeV, None),
            removeNeutral=True,
        ),
        outputDirCsv=outputDir,
        rnd=rnd,
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=oddDigiConfig,
        outputDirCsv=outputDir,
        rnd=rnd,
    )

    # make spacepoints from the measurements
    addSpacePointsMaking(s, trackingGeometry, oddSpacepointSel)

    # add spacepoint writer
    s.addWriter(
        acts.examples.CsvSpacepointWriter(
            inputSpacepoints = "spacepoints",
            outputDir = str(outputDir),
            level = acts.logging.INFO
        )
    )

    s.run()
