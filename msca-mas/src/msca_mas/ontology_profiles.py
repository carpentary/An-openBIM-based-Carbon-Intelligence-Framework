"""Ontology profiles used to resolve project queries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


MSCA = "http://bimlca.uni.edu/ontology#"
CPDS = "http://bimlca.uni.edu/cpd/schema#"
QUDT = "http://qudt.org/schema/qudt/"
PROV = "http://www.w3.org/ns/prov#"
RECON = "http://bimlca.uni.edu/reconstructed/station01/reference-case#"


class OntologyProfile(BaseModel):
    profile_id: str
    label: str
    description: str
    classes: list[str] = Field(default_factory=list)
    object_properties: list[str] = Field(default_factory=list)
    datatype_properties: list[str] = Field(default_factory=list)
    individuals: list[str] = Field(default_factory=list)
    extension_terms: list[str] = Field(default_factory=list)
    semantic_paths: list[dict[str, Any]] = Field(default_factory=list)
    source_modules: list[str] = Field(default_factory=list)

    @property
    def required_terms(self) -> list[str]:
        return sorted(
            set(
                self.classes
                + self.object_properties
                + self.datatype_properties
                + self.individuals
                + self.extension_terms
            )
        )


def ontology_profiles() -> dict[str, OntologyProfile]:
    profiles = [
        OntologyProfile(
            profile_id="ONT-PROJECT-CARBON",
            label="Project and life-cycle-stage carbon aggregation",
            description="Minimal signature for retrieving project totals and stage-resolved carbon emissions.",
            classes=[MSCA + "Project", MSCA + "CarbonAssessment", MSCA + "CarbonEmission", MSCA + "LifeCycleStage"],
            object_properties=[MSCA + "hasEmissions", MSCA + "belongsToStage", MSCA + "hasEmissionQuantity", QUDT + "unit"],
            datatype_properties=[QUDT + "numericValue"],
            individuals=[
                MSCA + "A1-A3_ProductStage",
                MSCA + "A4_TransportationStage",
                MSCA + "A5_ConstructionStage",
                MSCA + "Unit_KgCO2e",
            ],
            semantic_paths=[
                {
                    "path_id": "PATH-ASSESSMENT-STAGE-CARBON",
                    "triple_patterns": [
                        "?assessment rdf:type msca:CarbonAssessment",
                        "?assessment msca:hasEmissions ?emission",
                        "?emission msca:belongsToStage ?stage",
                        "?emission msca:hasEmissionQuantity ?quantity",
                        "?quantity qudt:numericValue ?value",
                        "?quantity qudt:unit msca:Unit_KgCO2e",
                    ],
                }
            ],
            source_modules=["ONTOLOGY_SCHEMA", "REFERENCE_ABOX", "REFERENCE_SUMMARY"],
        ),
        OntologyProfile(
            profile_id="ONT-SYSTEM-CARBON",
            label="Engineering-system S-E-P-R-C carbon path",
            description="Signature for separating main structure, enclosure and foundation-pit results.",
            classes=[
                MSCA + "EngineeringSystem",
                MSCA + "MainStructureSystem",
                MSCA + "EnclosureStructureSystem",
                MSCA + "FoundationPitSystem",
                MSCA + "PhysicalElement",
                MSCA + "ProjectWorkPackage",
                MSCA + "ResourceInstance",
                MSCA + "CarbonEmission",
            ],
            object_properties=[
                MSCA + "belongsToSystem",
                MSCA + "constructsElement",
                MSCA + "consumedByWorkPackage",
                MSCA + "calculatedFromResource",
                MSCA + "belongsToStage",
                MSCA + "hasEmissionQuantity",
                QUDT + "unit",
            ],
            datatype_properties=[QUDT + "numericValue"],
            individuals=[MSCA + "A1-A3_ProductStage", MSCA + "A4_TransportationStage", MSCA + "A5_ConstructionStage"],
            extension_terms=[RECON + "System_MainStructure", RECON + "System_Enclosure", RECON + "System_FoundationPit"],
            semantic_paths=[
                {
                    "path_id": "PATH-SYSTEM-SEPRC-CARBON",
                    "triple_patterns": [
                        "?element msca:belongsToSystem ?system",
                        "?workPackage msca:constructsElement ?element",
                        "?resource msca:consumedByWorkPackage ?workPackage",
                        "?emission msca:calculatedFromResource ?resource",
                        "?emission msca:belongsToStage ?stage",
                        "?emission msca:hasEmissionQuantity ?quantity",
                    ],
                }
            ],
            source_modules=["ONTOLOGY_SCHEMA", "CKG_SHARDS", "REFERENCE_ABOX", "REFERENCE_SUMMARY"],
        ),
        OntologyProfile(
            profile_id="ONT-ENCLOSURE-CPD",
            label="Enclosure element, work-package and CPD mapping",
            description="Signature for enclosure classification, quantities and CPD routing.",
            classes=[
                MSCA + "PhysicalElement",
                MSCA + "Type_DiaphragmWall",
                MSCA + "Type_MixedInPlacePile",
                MSCA + "ProjectWorkPackage",
                MSCA + "WorkPackageTemplate",
                MSCA + "CPD_Entry",
            ],
            object_properties=[
                MSCA + "belongsToSystem",
                MSCA + "constructsElement",
                MSCA + "requiresWorkPackage",
                MSCA + "instantiatedFrom",
                MSCA + "matchedFromCPD",
                MSCA + "hasQuantity",
                QUDT + "unit",
            ],
            datatype_properties=[MSCA + "hasGlobalID", QUDT + "numericValue"],
            extension_terms=[MSCA + "hasCPDSelectionDecision", CPDS + "CPDSelectionDecision", CPDS + "selectedOperationalEntry"],
            semantic_paths=[
                {
                    "path_id": "PATH-ENCLOSURE-ELEMENT-CPD",
                    "triple_patterns": [
                        "?element msca:belongsToSystem recon:System_Enclosure",
                        "?workPackage msca:constructsElement ?element",
                        "?workPackage msca:hasQuantity ?workQuantity",
                        "?workPackage msca:hasCPDSelectionDecision ?decision",
                        "?decision cpds:selectedOperationalEntry ?cpdEntry",
                    ],
                }
            ],
            source_modules=["ONTOLOGY_SCHEMA", "ENCLOSURE_ASSIGNMENTS", "REFERENCE_ABOX", "CPD_RECIPES"],
        ),
        OntologyProfile(
            profile_id="ONT-FOUNDATION-EARTHWORK",
            label="Foundation-pit earthwork quantities and emissions",
            description="Signature for excavation, disposal transport, backfill and compaction queries.",
            classes=[
                MSCA + "FoundationPitSystem",
                MSCA + "EarthworkZone",
                MSCA + "ExcavationZone",
                MSCA + "PhysicalElement",
                MSCA + "ProjectWorkPackage",
                MSCA + "ResourceInstance",
                MSCA + "CarbonEmission",
            ],
            object_properties=[
                MSCA + "belongsToSystem",
                MSCA + "hasQuantity",
                MSCA + "constructsElement",
                MSCA + "consumedByWorkPackage",
                MSCA + "calculatedFromResource",
                MSCA + "belongsToStage",
                MSCA + "hasEmissionQuantity",
            ],
            datatype_properties=[QUDT + "numericValue"],
            individuals=[MSCA + "PF_Earthwork", MSCA + "A1-A3_ProductStage", MSCA + "A4_TransportationStage", MSCA + "A5_ConstructionStage"],
            extension_terms=[RECON + "System_FoundationPit"],
            semantic_paths=[
                {
                    "path_id": "PATH-FOUNDATION-EARTHWORK-CARBON",
                    "triple_patterns": [
                        "?earthwork msca:belongsToSystem recon:System_FoundationPit",
                        "?workPackage msca:constructsElement ?earthwork",
                        "?resource msca:consumedByWorkPackage ?workPackage",
                        "?emission msca:calculatedFromResource ?resource",
                        "?emission msca:belongsToStage ?stage",
                    ],
                }
            ],
            source_modules=["ONTOLOGY_SCHEMA", "REFERENCE_PROFILE", "FOUNDATION_CPD_EFD", "REFERENCE_ABOX"],
        ),
        OntologyProfile(
            profile_id="ONT-A4-TRANSPORT",
            label="A4 material transportation",
            description="Signature for material, mass, distance, vehicle factor and A4 carbon queries.",
            classes=[MSCA + "TransportLeg", MSCA + "TransportMode", MSCA + "TruckTransport", MSCA + "ResourceInstance", MSCA + "CarbonEmission"],
            object_properties=[
                MSCA + "transportsResourceInstance",
                MSCA + "describedBy",
                MSCA + "hasTransportMode",
                MSCA + "travelDistance",
                MSCA + "hasEmissions",
                MSCA + "belongsToStage",
                MSCA + "hasEmissionQuantity",
            ],
            datatype_properties=[QUDT + "numericValue"],
            individuals=[MSCA + "A4_TransportationStage"],
            extension_terms=[RECON + "usesVehicle", RECON + "transportDistanceKm"],
            semantic_paths=[
                {
                    "path_id": "PATH-A4-TRANSPORT-LEG",
                    "triple_patterns": [
                        "?leg msca:transportsResourceInstance ?resource",
                        "?resource msca:describedBy ?descriptor",
                        "?leg msca:hasTransportMode ?mode",
                        "?leg msca:travelDistance ?distance",
                        "?leg msca:hasEmissions ?emission",
                        "?emission msca:belongsToStage msca:A4_TransportationStage",
                    ],
                },
                {
                    "path_id": "PATH-A4-REFERENCE-EXTENSION",
                    "triple_patterns": [
                        "?workPackage recon:usesVehicle ?vehicle",
                        "?workPackage recon:transportDistanceKm ?distanceKm",
                        "?resource msca:consumedByWorkPackage ?workPackage",
                        "?resource msca:hasEmissions ?emission",
                    ],
                },
            ],
            source_modules=["ONTOLOGY_SCHEMA", "A4_TRANSPORT_LEGS", "TRANSPORT_DEFAULTS", "REFERENCE_ABOX"],
        ),
        OntologyProfile(
            profile_id="ONT-SEPRC-PROVENANCE",
            label="System–Element–Process–Resource–Carbon provenance",
            description="Signature for auditable element-to-emission traces and factor provenance.",
            classes=[
                MSCA + "EngineeringSystem",
                MSCA + "PhysicalElement",
                MSCA + "ProjectWorkPackage",
                MSCA + "ResourceInstance",
                MSCA + "CarbonEmission",
                MSCA + "EFD_Factor",
                MSCA + "CarbonCalculationActivity",
            ],
            object_properties=[
                MSCA + "belongsToSystem",
                MSCA + "constructsElement",
                MSCA + "consumedByWorkPackage",
                MSCA + "describedBy",
                MSCA + "hasActualQuantity",
                MSCA + "calculatedFromResource",
                MSCA + "calculatedUsingFactor",
                MSCA + "hasEmissionQuantity",
                PROV + "wasGeneratedBy",
                PROV + "wasDerivedFrom",
            ],
            datatype_properties=[QUDT + "numericValue"],
            semantic_paths=[
                {
                    "path_id": "PATH-SEPRC-PROVENANCE",
                    "triple_patterns": [
                        "?element msca:belongsToSystem ?system",
                        "?workPackage msca:constructsElement ?element",
                        "?resource msca:consumedByWorkPackage ?workPackage",
                        "?resource msca:describedBy ?descriptor",
                        "?emission msca:calculatedFromResource ?resource",
                        "?emission msca:calculatedUsingFactor ?factor",
                        "?emission prov:wasGeneratedBy ?activity",
                    ],
                }
            ],
            source_modules=["ONTOLOGY_SCHEMA", "CKG_SHARDS", "REFERENCE_ABOX"],
        ),
        OntologyProfile(
            profile_id="ONT-SCENARIO",
            label="Transient scenario calculation",
            description="Signature for governed factor, material substitution and transport overlays.",
            classes=[MSCA + "AlternativeScenario", MSCA + "TransportOverride", MSCA + "ScenarioResult"],
            object_properties=[MSCA + "hasChangeSet", MSCA + "containsOverride", MSCA + "targetsEntity", MSCA + "alternativeTransportMode", MSCA + "alternativeDistance", MSCA + "hasEmissionDelta"],
            datatype_properties=[QUDT + "numericValue"],
            semantic_paths=[
                {
                    "path_id": "PATH-SCENARIO-OVERLAY-RESULT",
                    "triple_patterns": [
                        "?scenario msca:hasChangeSet ?changeSet",
                        "?changeSet msca:containsOverride ?override",
                        "?override msca:targetsEntity ?target",
                        "?result msca:hasEmissionDelta ?delta",
                    ],
                }
            ],
            source_modules=["ONTOLOGY_SCHEMA", "REFERENCE_SUMMARY"],
        ),
        OntologyProfile(
            profile_id="ONT-SCOPE-QUALITY",
            label="Assessment boundary and fail-closed quality",
            description="Signature for lifecycle-scope rejection and required-parameter quality checks.",
            classes=[MSCA + "CarbonAssessment", MSCA + "LifeCycleStage", MSCA + "QualityFlag"],
            object_properties=[MSCA + "belongsToStage", MSCA + "hasQualityFlag"],
            individuals=[
                MSCA + "A1-A3_ProductStage",
                MSCA + "A4_TransportationStage",
                MSCA + "A5_ConstructionStage",
                MSCA + "B1_Use",
                MSCA + "C1_Deconstruction",
                MSCA + "D_BeyondSystemBoundary",
            ],
            semantic_paths=[
                {
                    "path_id": "PATH-ASSESSMENT-SCOPE-CHECK",
                    "triple_patterns": [
                        "?assessment rdf:type msca:CarbonAssessment",
                        "?emission msca:belongsToStage ?requestedStage",
                        "FILTER(?requestedStage IN active A1-A3/A4/A5 stages)",
                    ],
                }
            ],
            source_modules=["ONTOLOGY_SCHEMA", "PROJECT_CONFIG"],
        ),
    ]
    return {profile.profile_id: profile for profile in profiles}
