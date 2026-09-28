// Type definitions for car_info3.json / car_info3_vector.json
// 차량 한 대를 9개 섹션 레코드로 표현하는 구조에 대한 타입 정의

// 공통 베이스 타입
export interface CarRecordBase {
  id: string;
  vector: number[]; // OpenAI text-embedding-3-small (1536차원)
  payload: CarPayload;
}

// 공통 payload 베이스
export interface CarPayloadBase {
  car_id: string;
  section: CarSection;
  nl: string;
}

export type CarSection =
  | "summary"
  | "basic_info"
  | "special_usage_history"
  | "special_accident_history"
  | "registration_change_history"
  | "insurance_accident_history"
  | "options"
  | "seller_info"
  | "inspection_record";

// 섹션별 raw 타입 정의

export interface SummaryRaw {
  summary_text: string;
}

export interface BasicInfoRaw {
  model_name: string;
  manufacturer: string;
  first_registration_date: string;
  vin: string;
  transmission_type: string;
  exterior_color: string;
  interior_color: string;
  displacement_cc: number;
  engine_type: string;
  usage_type: string;
  fuel_type: string;
  first_insurance_date: string;
  year: number;
  mileage_km: number;
  mileage_status: string;
}

export interface SpecialUsageHistoryRaw {
  rental_car: boolean;
  taxi: boolean;
  government_use: boolean;
  description: string;
}

export interface SpecialAccidentHistoryItem {
  occurred: boolean;
  description: string;
}

export interface SpecialAccidentHistoryRaw {
  total_loss: SpecialAccidentHistoryItem;
  theft: SpecialAccidentHistoryItem;
  flood: SpecialAccidentHistoryItem;
}

export interface RegistrationChangeHistoryItem {
  change_date: string;
  plate_number: string;
  owner_change: boolean;
  usage_type: string;
}

export type RegistrationChangeHistoryRaw = RegistrationChangeHistoryItem[];

export interface MyCarDamage {
  occurred: boolean;
  repair_estimate_krw: number;
  parts_cost_krw: number;
  labor_cost_krw: number;
  painting_cost_krw: number;
  damaged_parts: string[];
  repair_details: string;
}

export interface OpponentInsurance {
  processed: boolean;
  property_damage: string;
}

export interface InsuranceAccidentHistoryItem {
  accident_date: string;
  my_car_damage: MyCarDamage;
  opponent_insurance: OpponentInsurance;
  my_insurance_processed: boolean;
}

export type InsuranceAccidentHistoryRaw = InsuranceAccidentHistoryItem[];

// options 섹션

export interface SideMirrorOptions {
  manual: boolean;
  heated: boolean;
  auto_angle_adjustment: boolean;
  power_folding: boolean;
}

export interface SunroofOptions {
  standard: boolean;
  dual: boolean;
  panoramic: boolean;
}

export interface WindshieldOptions {
  wiper_defrost: boolean;
  uv_protection: boolean;
  rain_sensor_wiper: boolean;
}

export interface WheelOptions {
  steel: boolean;
  aluminum: boolean;
  chrome: boolean;
  wide_tire: boolean;
}

export interface HeadlampOptions {
  HID: boolean;
  LED: boolean;
  laser_light: boolean;
  adaptive: boolean;
  high_beam_assist: boolean;
  halogen: boolean;
  projection_type: boolean;
}

export interface ExteriorOptions {
  roof_rack: boolean;
  integrated_turn_signal: boolean;
  side_mirror: SideMirrorOptions;
  sunroof: SunroofOptions;
  windshield: WindshieldOptions;
  wheels: WheelOptions;
  headlamp: HeadlampOptions;
}

export interface RearviewMirrorOptions {
  electronic_ecm: boolean;
  hipass_integrated: boolean;
  rear_view_mirror: boolean;
}

export interface PowerSeatsOptions {
  driver: boolean;
  passenger: boolean;
  rear: boolean;
}

export interface HeatedSeatsOptions {
  front: boolean;
  rear: boolean;
}

export interface VentilatedSeatsOptions {
  driver: boolean;
  passenger: boolean;
}

export interface MemorySeatsOptions {
  driver: boolean;
  passenger: boolean;
}

export interface SeatsOptions {
  material: string;
  power_seats: PowerSeatsOptions;
  heated_seats: HeatedSeatsOptions;
  ventilated_seats: VentilatedSeatsOptions;
  memory_seats: MemorySeatsOptions;
  massage_seat: boolean;
}

export interface SteeringWheelOptions {
  material: string;
  heated: boolean;
  telescopic: boolean;
  speed_sensitive: boolean;
}

export interface ParkingBrakeOptions {
  type: string;
}

export interface InteriorOptions {
  rearview_mirror: RearviewMirrorOptions;
  seats: SeatsOptions;
  steering_wheel: SteeringWheelOptions;
  parking_brake: ParkingBrakeOptions;
}

export interface AirbagsOptions {
  driver: boolean;
  passenger: boolean;
  side: boolean;
  curtain: boolean;
  knee: boolean;
  total_count: number;
}

export interface DrivingControlOptions {
  ABS: boolean;
  ESC_VDC_ESP: boolean;
  TCS: boolean;
  VSM: boolean;
  EBD: boolean;
  ESS: boolean;
}

export interface SensorsCamerasOptions {
  front_sensor: boolean;
  rear_sensor: boolean;
  front_camera: boolean;
  rear_camera: boolean;
  around_view_monitor: boolean;
}

export interface AdvancedAssistanceOptions {
  BSD: boolean;
  LDWS: boolean;
  LKAS: boolean;
  FCW: boolean;
  AEB: boolean;
  HAS: boolean;
  TPMS: boolean;
  head_up_display: boolean;
}

export interface PassiveSafetyOptions {
  isofix: boolean;
  safety_window: boolean;
  active_headrest: boolean;
}

export interface SafetyOptions {
  airbags: AirbagsOptions;
  driving_control: DrivingControlOptions;
  sensors_cameras: SensorsCamerasOptions;
  advanced_assistance: AdvancedAssistanceOptions;
  passive_safety: PassiveSafetyOptions;
}

export interface AirConditioningOptions {
  type: string;
  air_purifier: boolean;
}

export interface EntryStartOptions {
  smart_key: boolean;
  button_start: boolean;
}

export interface DrivingAidsOptions {
  steering_wheel_remote: boolean;
  power_trunk: boolean;
  auto_parking_assist: boolean;
  cruise_control: string;
  hands_free: boolean;
}

export interface MultimediaOptions {
  navigation: string;
  bluetooth: boolean;
  AUX: boolean;
  USB: boolean;
  speaker: string;
  CD: boolean;
  CD_changer: boolean;
  DVD: boolean;
  MP3: boolean;
}

export interface ConvenienceOptions {
  air_conditioning: AirConditioningOptions;
  entry_start: EntryStartOptions;
  driving_aids: DrivingAidsOptions;
  multimedia: MultimediaOptions;
}

export interface OptionsRaw {
  exterior: ExteriorOptions;
  interior: InteriorOptions;
  safety: SafetyOptions;
  convenience: ConvenienceOptions;
}

export interface SellerInfoRaw {
  dealer_name: string;
  phone_number: string;
  recommendation: string;
}

export interface OverallCondition {
  mileage_tampering: string;
  emission: {
    carbon_monoxide: string;
    hydrocarbon: string;
    smoke: string;
  };
  tuning_history: {
    exists: boolean;
    legal_status: string;
  };
}

export interface ExteriorPanelsState {
  [panelName: string]: string;
}

export interface MainFrameState {
  [partName: string]: string;
}

export interface AccidentRepairHistoryRaw {
  exterior_panels: ExteriorPanelsState;
  main_frame: MainFrameState;
}

export interface InspectionRecordRaw {
  overall_condition: OverallCondition;
  accident_repair_history: AccidentRepairHistoryRaw;
}

// 섹션별 payload 타입

export interface SummaryPayload extends CarPayloadBase {
  section: "summary";
  raw: SummaryRaw;
}

export interface BasicInfoPayload extends CarPayloadBase {
  section: "basic_info";
  raw: BasicInfoRaw;
}

export interface SpecialUsageHistoryPayload extends CarPayloadBase {
  section: "special_usage_history";
  raw: SpecialUsageHistoryRaw;
}

export interface SpecialAccidentHistoryPayload extends CarPayloadBase {
  section: "special_accident_history";
  raw: SpecialAccidentHistoryRaw;
}

export interface RegistrationChangeHistoryPayload extends CarPayloadBase {
  section: "registration_change_history";
  raw: RegistrationChangeHistoryRaw;
}

export interface InsuranceAccidentHistoryPayload extends CarPayloadBase {
  section: "insurance_accident_history";
  raw: InsuranceAccidentHistoryRaw;
}

export interface OptionsPayload extends CarPayloadBase {
  section: "options";
  raw: OptionsRaw;
}

export interface SellerInfoPayload extends CarPayloadBase {
  section: "seller_info";
  raw: SellerInfoRaw;
}

export interface InspectionRecordPayload extends CarPayloadBase {
  section: "inspection_record";
  raw: InspectionRecordRaw;
}

export type CarPayload =
  | SummaryPayload
  | BasicInfoPayload
  | SpecialUsageHistoryPayload
  | SpecialAccidentHistoryPayload
  | RegistrationChangeHistoryPayload
  | InsuranceAccidentHistoryPayload
  | OptionsPayload
  | SellerInfoPayload
  | InspectionRecordPayload;

export type CarRecord = CarRecordBase;


