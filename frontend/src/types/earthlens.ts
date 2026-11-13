export interface StudyArea {
  id: string;
  name: string;
  description: string;
  center: [number, number]; // [lon, lat]
  zoom: number;
  bbox: [number, number, number, number];
  dates: {
    before: string;
    after: string;
  };
  attribution: string;
}

export interface LandCoverDistribution {
  classId: number;
  name: string;
  pixelCount: number;
  areaKm2: number;
  percentage: number;
  color: string;
}

export interface CategoryChange {
  classId: number;
  name: string;
  beforeAreaKm2: number;
  afterAreaKm2: number;
  netChangeKm2: number;
  percentChange: number;
  color: string;
}

export interface AnalysisSummary {
  totalStudyAreaKm2: number;
  totalChangedAreaKm2: number;
  changePercentage: number;
  waterLossKm2: number;
  urbanGainKm2: number;
  vegetationLossKm2: number;
  keyDriver: string;
}

export interface ModelMetrics {
  accuracy: number;
  f1_macro: number;
  classes: Record<string, { precision: number; recall: number; f1: number }>;
  confusion_matrix: number[][];
  n_estimators: number;
  features: string[];
}

export interface ChangeAnalysisResponse {
  studyArea: StudyArea;
  summary: AnalysisSummary;
  distributionBefore: LandCoverDistribution[];
  distributionAfter: LandCoverDistribution[];
  categoryChanges: CategoryChange[];
  images: {
    satelliteBefore: string;
    satelliteAfter: string;
    classificationBefore: string;
    classificationAfter: string;
    changeOverlay: string;
  };
  modelMetrics: ModelMetrics;
}
