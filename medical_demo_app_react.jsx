import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import {
  Search,
  Plus,
  Download,
  Upload,
  Printer,
  Sun,
  Moon,
  Stethoscope,
  FileEdit,
  Trash2,
  Save,
  UserPlus,
  HeartPulse,
  History,
  ShieldAlert,
  Pill,
} from "lucide-react";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// ----------------- Types -----------------

type Sex = "Male" | "Female" | "Other";

interface Encounter {
  id: string;
  date: string; // ISO string
  notes: string;
  systolic?: number;
  diastolic?: number;
  heartRate?: number;
  tempC?: number;
  spo2?: number;
  weightKg?: number;
  heightCm?: number;
  diagnosis?: string;
  prescription?: string;
}

interface Patient {
  id: string;
  mrn: string;
  firstName: string;
  lastName: string;
  dob: string; // ISO date
  sex: Sex;
  phone?: string;
  email?: string;
  address?: string;
  allergies: string[];
  conditions: string[];
  meds: string[];
  heightCm?: number;
  weightKg?: number;
  notes?: string;
  encounters: Encounter[];
}

// ----------------- Utilities -----------------

const uid = () => Math.random().toString(36).slice(2, 10);
const generateMRN = () => `${Date.now().toString().slice(-6)}-${Math
  .floor(Math.random() * 900 + 100)
  .toString()}`;

function calcAge(dobIso: string) {
  const dob = new Date(dobIso);
  const now = new Date();
  let age = now.getFullYear() - dob.getFullYear();
  const m = now.getMonth() - dob.getMonth();
  if (m < 0 || (m === 0 && now.getDate() < dob.getDate())) age--;
  return age;
}

function bmi(heightCm?: number, weightKg?: number) {
  if (!heightCm || !weightKg) return undefined;
  const m = heightCm / 100;
  return +(weightKg / (m * m)).toFixed(1);
}

function bmiCategory(b?: number) {
  if (b == null) return "";
  if (b < 18.5) return "Underweight";
  if (b < 25) return "Normal";
  if (b < 30) return "Overweight";
  return "Obese";
}

function toTitle(s: string) {
  return s.replace(/\b\w/g, (c) => c.toUpperCase());
}

function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ----------------- Seed Data -----------------

const seedPatients: Patient[] = [
  {
    id: uid(),
    mrn: generateMRN(),
    firstName: "Ava",
    lastName: "Ng",
    dob: "1993-05-14",
    sex: "Female",
    phone: "+27 72 555 0123",
    email: "ava.ng@example.com",
    address: "12 Loop St, Cape Town",
    allergies: ["Penicillin"],
    conditions: ["Hypertension"],
    meds: ["Amlodipine 5mg OD"],
    heightCm: 164,
    weightKg: 68,
    notes: "Prefers afternoon appointments.",
    encounters: [
      {
        id: uid(),
        date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 120).toISOString(),
        notes: "Routine check-up. BP elevated.",
        systolic: 142,
        diastolic: 90,
        heartRate: 82,
        tempC: 36.7,
        spo2: 98,
        diagnosis: "Hypertension",
        prescription: "Amlodipine 5mg OD",
        weightKg: 69,
        heightCm: 164,
      },
      {
        id: uid(),
        date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
        notes: "Follow-up. BP improved.",
        systolic: 128,
        diastolic: 82,
        heartRate: 76,
        tempC: 36.5,
        spo2: 99,
        diagnosis: "Hypertension (controlled)",
        prescription: "Amlodipine 5mg OD",
        weightKg: 68,
        heightCm: 164,
      },
    ],
  },
  {
    id: uid(),
    mrn: generateMRN(),
    firstName: "Thabo",
    lastName: "Dlamini",
    dob: "1981-11-02",
    sex: "Male",
    phone: "+27 83 111 4545",
    email: "thabo.d@example.com",
    address: "45 Oxford Rd, Johannesburg",
    allergies: [],
    conditions: ["Type 2 Diabetes"],
    meds: ["Metformin 1g BD"],
    heightCm: 176,
    weightKg: 92,
    notes: "Brings home glucose logs.",
    encounters: [
      {
        id: uid(),
        date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 10).toISOString(),
        notes: "Discuss diet. HbA1c pending.",
        heartRate: 78,
        tempC: 36.6,
        spo2: 98,
        weightKg: 92,
        heightCm: 176,
        diagnosis: "T2DM",
        prescription: "Continue Metformin",
      },
    ],
  },
];

// ----------------- Patient Form -----------------

function ChipEditor({
  label,
  items,
  setItems,
  placeholder,
}: {
  label: string;
  items: string[];
  setItems: (next: string[]) => void;
  placeholder?: string;
}) {
  const [value, setValue] = useState("");
  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">{label}</Label>
      <div className="flex gap-2 items-center">
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder || `Add ${label.toLowerCase()}`}
          onKeyDown={(e) => {
            if (e.key === "Enter" && value.trim()) {
              setItems([...items, value.trim()]);
              setValue("");
            }
          }}
        />
        <Button
          type="button"
          variant="secondary"
          onClick={() => {
            if (!value.trim()) return;
            setItems([...items, value.trim()]);
            setValue("");
          }}
        >
          Add
        </Button>
      </div>
      <div className="flex flex-wrap gap-2">
        {items.map((t, i) => (
          <Badge
            key={`${t}-${i}`}
            variant="secondary"
            className="cursor-pointer"
            onClick={() => setItems(items.filter((_, idx) => idx !== i))}
            title="Click to remove"
          >
            {t}
          </Badge>
        ))}
      </div>
    </div>
  );
}

function PatientForm({
  initial,
  onSave,
  onCancel,
}: {
  initial?: Partial<Patient>;
  onSave: (p: Patient) => void;
  onCancel?: () => void;
}) {
  const [firstName, setFirstName] = useState(initial?.firstName || "");
  const [lastName, setLastName] = useState(initial?.lastName || "");
  const [dob, setDob] = useState(initial?.dob || "");
  const [sex, setSex] = useState<Sex>((initial?.sex as Sex) || "Male");
  const [phone, setPhone] = useState(initial?.phone || "");
  const [email, setEmail] = useState(initial?.email || "");
  const [address, setAddress] = useState(initial?.address || "");
  const [allergies, setAllergies] = useState<string[]>(initial?.allergies || []);
  const [conditions, setConditions] = useState<string[]>(
    initial?.conditions || []
  );
  const [meds, setMeds] = useState<string[]>(initial?.meds || []);
  const [heightCm, setHeightCm] = useState<number | undefined>(
    initial?.heightCm
  );
  const [weightKg, setWeightKg] = useState<number | undefined>(
    initial?.weightKg
  );
  const [notes, setNotes] = useState(initial?.notes || "");

  const b = bmi(heightCm, weightKg);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!firstName.trim() || !lastName.trim() || !dob) return;
    const p: Patient = {
      id: initial?.id || uid(),
      mrn: initial?.mrn || generateMRN(),
      firstName: toTitle(firstName.trim()),
      lastName: toTitle(lastName.trim()),
      dob,
      sex,
      phone: phone.trim() || undefined,
      email: email.trim() || undefined,
      address: address.trim() || undefined,
      allergies,
      conditions,
      meds,
      heightCm,
      weightKg,
      notes: notes.trim() || undefined,
      encounters: initial?.encounters || [],
    };
    onSave(p);
  }

  return (
    <form className="grid gap-4" onSubmit={handleSubmit}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>First name *</Label>
          <Input value={firstName} onChange={(e) => setFirstName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label>Last name *</Label>
          <Input value={lastName} onChange={(e) => setLastName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label>Date of birth *</Label>
          <Input type="date" value={dob} onChange={(e) => setDob(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label>Sex</Label>
          <Select value={sex} onValueChange={(v) => setSex(v as Sex)}>
            <SelectTrigger>
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Male">Male</SelectItem>
              <SelectItem value="Female">Female</SelectItem>
              <SelectItem value="Other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label>Phone</Label>
          <Input value={phone} onChange={(e) => setPhone(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label>Email</Label>
          <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div className="space-y-2 md:col-span-2">
          <Label>Address</Label>
          <Input value={address} onChange={(e) => setAddress(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label>Height (cm)</Label>
          <Input
            type="number"
            inputMode="decimal"
            value={heightCm ?? ""}
            onChange={(e) => setHeightCm(e.target.value ? Number(e.target.value) : undefined)}
          />
        </div>
        <div className="space-y-2">
          <Label>Weight (kg)</Label>
          <Input
            type="number"
            inputMode="decimal"
            value={weightKg ?? ""}
            onChange={(e) => setWeightKg(e.target.value ? Number(e.target.value) : undefined)}
          />
        </div>
        <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-3 gap-4">
          <ChipEditor
            label="Allergies"
            items={allergies}
            setItems={setAllergies}
            placeholder="e.g., Penicillin"
          />
          <ChipEditor
            label="Conditions"
            items={conditions}
            setItems={setConditions}
            placeholder="e.g., Hypertension"
          />
          <ChipEditor
            label="Medications"
            items={meds}
            setItems={setMeds}
            placeholder="e.g., Metformin 1g BD"
          />
        </div>
        <div className="md:col-span-2 space-y-2">
          <Label>Clinical notes</Label>
          <Textarea rows={4} value={notes} onChange={(e) => setNotes(e.target.value)} />
        </div>
      </div>

      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <HeartPulse className="w-4 h-4" />
        <span>
          BMI: <span className="font-medium">{b ?? "—"}</span>{" "}
          {b != null && <span className="ml-1">({bmiCategory(b)})</span>}
        </span>
      </div>

      <div className="flex gap-2">
        <Button type="submit" className="gap-2">
          <Save className="w-4 h-4" /> Save
        </Button>
        {onCancel && (
          <Button type="button" variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>
    </form>
  );
}

// ----------------- Encounter Dialog -----------------

function EncounterDialog({
  onAdd,
  trigger,
  defaultHeight,
  defaultWeight,
}: {
  onAdd: (e: Encounter) => void;
  trigger?: React.ReactNode;
  defaultHeight?: number;
  defaultWeight?: number;
}) {
  const [open, setOpen] = useState(false);
  const [date, setDate] = useState<string>(new Date().toISOString().slice(0, 16));
  const [notes, setNotes] = useState("");
  const [systolic, setSystolic] = useState<number | undefined>();
  const [diastolic, setDiastolic] = useState<number | undefined>();
  const [heartRate, setHeartRate] = useState<number | undefined>();
  const [tempC, setTempC] = useState<number | undefined>();
  const [spo2, setSpo2] = useState<number | undefined>();
  const [weightKg, setWeightKg] = useState<number | undefined>(defaultWeight);
  const [heightCm, setHeightCm] = useState<number | undefined>(defaultHeight);
  const [diagnosis, setDiagnosis] = useState<string>("");
  const [prescription, setPrescription] = useState<string>("");

  function addEncounter() {
    const enc: Encounter = {
      id: uid(),
      date: new Date(date).toISOString(),
      notes: notes.trim(),
      systolic,
      diastolic,
      heartRate,
      tempC,
      spo2,
      weightKg,
      heightCm,
      diagnosis: diagnosis.trim() || undefined,
      prescription: prescription.trim() || undefined,
    };
    onAdd(enc);
    setOpen(false);
    // reset minimal fields
    setNotes("");
    setDiagnosis("");
    setPrescription("");
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger || <Button>Add encounter</Button>}</DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <History className="w-5 h-5" /> New Encounter
          </DialogTitle>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Date & time</Label>
              <Input
                type="datetime-local"
                value={date}
                onChange={(e) => setDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Temperature (°C)</Label>
              <Input
                type="number"
                inputMode="decimal"
                value={tempC ?? ""}
                onChange={(e) => setTempC(e.target.value ? Number(e.target.value) : undefined)}
              />
            </div>
            <div className="space-y-2">
              <Label>Heart rate (bpm)</Label>
              <Input
                type="number"
                inputMode="numeric"
                value={heartRate ?? ""}
                onChange={(e) => setHeartRate(e.target.value ? Number(e.target.value) : undefined)}
              />
            </div>
            <div className="space-y-2">
              <Label>SpO₂ (%)</Label>
              <Input
                type="number"
                inputMode="numeric"
                value={spo2 ?? ""}
                onChange={(e) => setSpo2(e.target.value ? Number(e.target.value) : undefined)}
              />
            </div>
            <div className="space-y-2">
              <Label>Blood pressure (mmHg)</Label>
              <div className="flex gap-2">
                <Input
                  placeholder="Systolic"
                  type="number"
                  inputMode="numeric"
                  value={systolic ?? ""}
                  onChange={(e) => setSystolic(e.target.value ? Number(e.target.value) : undefined)}
                />
                <Input
                  placeholder="Diastolic"
                  type="number"
                  inputMode="numeric"
                  value={diastolic ?? ""}
                  onChange={(e) => setDiastolic(e.target.value ? Number(e.target.value) : undefined)}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Weight (kg)</Label>
              <Input
                type="number"
                inputMode="decimal"
                value={weightKg ?? ""}
                onChange={(e) => setWeightKg(e.target.value ? Number(e.target.value) : undefined)}
              />
            </div>
            <div className="space-y-2">
              <Label>Height (cm)</Label>
              <Input
                type="number"
                inputMode="decimal"
                value={heightCm ?? ""}
                onChange={(e) => setHeightCm(e.target.value ? Number(e.target.value) : undefined)}
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label>Diagnosis</Label>
            <Input value={diagnosis} onChange={(e) => setDiagnosis(e.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>Prescription</Label>
            <Input value={prescription} onChange={(e) => setPrescription(e.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>Notes</Label>
            <Textarea rows={4} value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
        </div>
        <DialogFooter>
          <Button className="gap-2" onClick={addEncounter}>
            <Plus className="w-4 h-4" /> Add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ----------------- Patient Detail View -----------------

function PatientDetail({
  patient,
  onUpdate,
  onDelete,
}: {
  patient: Patient;
  onUpdate: (p: Patient) => void;
  onDelete: (id: string) => void;
}) {
  const [mode, setMode] = useState<"view" | "edit">("view");

  const vitalsForChart = useMemo(() => {
    return patient.encounters
      .slice()
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .map((e) => ({
        date: new Date(e.date).toLocaleDateString(),
        heartRate: e.heartRate ?? null,
        systolic: e.systolic ?? null,
        diastolic: e.diastolic ?? null,
        weightKg: e.weightKg ?? null,
      }));
  }, [patient.encounters]);

  if (mode === "edit") {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileEdit className="w-5 h-5" /> Edit Patient
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <PatientForm
            initial={patient}
            onSave={(p) => {
              onUpdate(p);
              setMode("view");
            }}
            onCancel={() => setMode("view")}
          />
        </CardContent>
      </Card>
    );
  }

  const b = bmi(patient.heightCm, patient.weightKg);

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
            <Stethoscope className="w-5 h-5 text-primary" />
          </div>
          <div>
            <div className="text-xl font-semibold">
              {patient.firstName} {patient.lastName}
            </div>
            <div className="text-sm text-muted-foreground">
              MRN {patient.mrn} • {calcAge(patient.dob)}y • {patient.sex}
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" className="gap-2">
                <Download className="w-4 h-4" /> Export JSON
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Export patient record</DialogTitle>
              </DialogHeader>
              <p className="text-sm text-muted-foreground">
                Export this patient as a JSON file. Suitable for demos or importing
                back into this app.
              </p>
              <div className="flex justify-end">
                <Button
                  className="gap-2"
                  onClick={() => downloadJson(`patient-${patient.mrn}.json`, patient)}
                >
                  <Download className="w-4 h-4" /> Download
                </Button>
              </div>
            </DialogContent>
          </Dialog>
          <Button variant="outline" className="gap-2" onClick={() => window.print()}>
            <Printer className="w-4 h-4" /> Print
          </Button>
          <Button className="gap-2" onClick={() => setMode("edit")}> 
            <FileEdit className="w-4 h-4" /> Edit
          </Button>
          <Button
            variant="destructive"
            className="gap-2"
            onClick={() => onDelete(patient.id)}
          >
            <Trash2 className="w-4 h-4" /> Delete
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
            <CardDescription>Demographics & clinical snapshot</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-muted-foreground">DOB</div>
              <div>{new Date(patient.dob).toLocaleDateString()}</div>

              <div className="text-muted-foreground">Phone</div>
              <div>{patient.phone || "—"}</div>

              <div className="text-muted-foreground">Email</div>
              <div>{patient.email || "—"}</div>

              <div className="text-muted-foreground">Address</div>
              <div>{patient.address || "—"}</div>

              <div className="text-muted-foreground">Height</div>
              <div>{patient.heightCm ? `${patient.heightCm} cm` : "—"}</div>

              <div className="text-muted-foreground">Weight</div>
              <div>{patient.weightKg ? `${patient.weightKg} kg` : "—"}</div>

              <div className="text-muted-foreground">BMI</div>
              <div>
                {b ?? "—"} {b != null && <span className="text-muted-foreground">({bmiCategory(b)})</span>}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Allergies</div>
              <div className="flex flex-wrap gap-2">
                {patient.allergies.length ? (
                  patient.allergies.map((a, i) => (
                    <Badge key={i} variant="secondary">{a}</Badge>
                  ))
                ) : (
                  <span className="text-sm">None</span>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Conditions</div>
              <div className="flex flex-wrap gap-2">
                {patient.conditions.length ? (
                  patient.conditions.map((c, i) => (
                    <Badge key={i} variant="secondary">{c}</Badge>
                  ))
                ) : (
                  <span className="text-sm">None</span>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Medications</div>
              <div className="flex flex-wrap gap-2">
                {patient.meds.length ? (
                  patient.meds.map((m, i) => (
                    <Badge key={i} variant="outline" className="flex items-center gap-1">
                      <Pill className="w-3 h-3" /> {m}
                    </Badge>
                  ))
                ) : (
                  <span className="text-sm">None</span>
                )}
              </div>
            </div>

            {patient.notes && (
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Notes</div>
                <div className="text-sm leading-relaxed">{patient.notes}</div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Vitals over time</CardTitle>
            <CardDescription>Trend of selected measurements</CardDescription>
          </CardHeader>
          <CardContent>
            {vitalsForChart.length ? (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={vitalsForChart} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="heartRate" name="HR (bpm)" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="systolic" name="Sys" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="diastolic" name="Dia" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="weightKg" name="Weight (kg)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">No encounters with vitals yet.</div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="w-5 h-5" /> Encounters
          </CardTitle>
          <CardDescription>Most recent first</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex justify-end mb-3">
            <EncounterDialog
              defaultHeight={patient.heightCm}
              defaultWeight={patient.weightKg}
              onAdd={(enc) =>
                onUpdate({ ...patient, encounters: [enc, ...patient.encounters] })
              }
              trigger={
                <Button className="gap-2">
                  <Plus className="w-4 h-4" /> Add encounter
                </Button>
              }
            />
          </div>
          {patient.encounters.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>BP</TableHead>
                  <TableHead>HR</TableHead>
                  <TableHead>Temp</TableHead>
                  <TableHead>SpO₂</TableHead>
                  <TableHead>Diagnosis</TableHead>
                  <TableHead>Prescription</TableHead>
                  <TableHead>Notes</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {patient.encounters
                  .slice()
                  .sort(
                    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
                  )
                  .map((e) => (
                    <TableRow key={e.id}>
                      <TableCell className="whitespace-nowrap">
                        {new Date(e.date).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        {e.systolic && e.diastolic ? `${e.systolic}/${e.diastolic}` : "—"}
                      </TableCell>
                      <TableCell>{e.heartRate ?? "—"}</TableCell>
                      <TableCell>{e.tempC ? `${e.tempC}°C` : "—"}</TableCell>
                      <TableCell>{e.spo2 ? `${e.spo2}%` : "—"}</TableCell>
                      <TableCell>{e.diagnosis || "—"}</TableCell>
                      <TableCell>{e.prescription || "—"}</TableCell>
                      <TableCell className="max-w-[24rem] truncate" title={e.notes}>
                        {e.notes || "—"}
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-sm text-muted-foreground">No encounters yet.</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ----------------- Main App -----------------

export default function MedicalDemoApp() {
  const [patients, setPatients] = useState<Patient[]>(() => seedPatients);
  const [selectedId, setSelectedId] = useState<string | null>(
    seedPatients[0]?.id || null
  );
  const [query, setQuery] = useState("");
  const [persist, setPersist] = useState<boolean>(() => {
    const saved = localStorage.getItem("demo_persist");
    return saved ? saved === "1" : false;
  });
  const [dark, setDark] = useState<boolean>(() => {
    const saved = localStorage.getItem("demo_dark");
    return saved ? saved === "1" : false;
  });
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // persistence
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("demo_dark", dark ? "1" : "0");
  }, [dark]);

  useEffect(() => {
    if (persist) {
      const saved = localStorage.getItem("demo_patients");
      if (saved) {
        try {
          const parsed = JSON.parse(saved) as Patient[];
          setPatients(parsed);
          setSelectedId(parsed[0]?.id || null);
        } catch {}
      }
    }
  }, []); // run once on mount

  useEffect(() => {
    if (persist) {
      localStorage.setItem("demo_patients", JSON.stringify(patients));
      localStorage.setItem("demo_persist", "1");
    } else {
      localStorage.removeItem("demo_patients");
      localStorage.setItem("demo_persist", "0");
    }
  }, [patients, persist]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((p) => {
      const hay = [
        p.firstName,
        p.lastName,
        p.mrn,
        p.phone || "",
        p.email || "",
        ...(p.conditions || []),
        ...(p.meds || []),
      ]
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
  }, [patients, query]);

  const selected = patients.find((p) => p.id === selectedId) || null;

  function addPatient(p: Patient) {
    setPatients([p, ...patients]);
    setSelectedId(p.id);
  }

  function updatePatient(p: Patient) {
    setPatients((arr) => arr.map((x) => (x.id === p.id ? p : x)));
  }

  function deletePatient(id: string) {
    setPatients((arr) => arr.filter((x) => x.id !== id));
    if (selectedId === id) setSelectedId(null);
  }

  function importFromFile(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));
        if (Array.isArray(data)) {
          // import array of patients
          const normalized = (data as Patient[]).map((p) => ({ ...p, id: p.id || uid() }));
          setPatients((prev) => [...normalized, ...prev]);
        } else {
          const p = data as Patient;
          p.id = p.id || uid();
          setPatients((prev) => [p, ...prev]);
        }
      } catch (e) {
        alert("Invalid JSON");
      }
    };
    reader.readAsText(file);
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-7xl mx-auto p-4 md:p-6 lg:p-8 print:p-0">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4 print:hidden">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-primary/10 flex items-center justify-center shadow-sm">
              <Stethoscope className="w-5 h-5 text-primary" />
            </div>
            <div>
              <div className="text-2xl font-bold tracking-tight">
                Medical Demo App
              </div>
              <div className="text-sm text-muted-foreground">
                In-memory EMR-style demo — no backend
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm">
              <Switch checked={persist} onCheckedChange={setPersist} id="persist" />
              <Label htmlFor="persist">Persist locally</Label>
            </div>
            <Button variant="outline" className="gap-2" onClick={() => setDark((d) => !d)}>
              {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />} Theme
            </Button>
          </div>
        </div>

        {/* Banner */}
        <Card className="mb-4 border-destructive/30 print:hidden">
          <CardContent className="py-3 flex items-center gap-3 text-sm">
            <ShieldAlert className="w-4 h-4 text-destructive" />
            <span>
              Demo only — do <span className="font-semibold">not</span> enter real
              patient identifiers or PHI.
            </span>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 print:hidden">
            <Card className="sticky top-4">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Patients</CardTitle>
                <CardDescription>Search, select, or add</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <div className="relative w-full">
                    <Search className="w-4 h-4 absolute left-2 top-2.5 text-muted-foreground" />
                    <Input
                      className="pl-8"
                      placeholder="Search name, MRN, phone, med..."
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                    />
                  </div>
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button className="gap-2" title="Add patient">
                        <UserPlus className="w-4 h-4" /> New
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-3xl">
                      <DialogHeader>
                        <DialogTitle>Add patient</DialogTitle>
                      </DialogHeader>
                      <PatientForm onSave={(p) => addPatient(p)} />
                    </DialogContent>
                  </Dialog>
                </div>

                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    className="gap-2"
                    onClick={() => downloadJson("patients.json", patients)}
                  >
                    <Download className="w-4 h-4" /> Export all
                  </Button>
                  <Button
                    variant="outline"
                    className="gap-2"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="w-4 h-4" /> Import
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/json"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) importFromFile(f);
                      e.currentTarget.value = "";
                    }}
                  />
                </div>

                <div className="max-h-[50vh] overflow-auto rounded-md border">
                  {filtered.length ? (
                    <Table>
                      <TableBody>
                        {filtered.map((p) => (
                          <TableRow
                            key={p.id}
                            className={`cursor-pointer ${
                              p.id === selectedId ? "bg-muted" : ""
                            }`}
                            onClick={() => setSelectedId(p.id)}
                          >
                            <TableCell>
                              <div className="font-medium">
                                {p.firstName} {p.lastName}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                MRN {p.mrn} • {calcAge(p.dob)}y • {p.sex}
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <div className="p-4 text-sm text-muted-foreground">
                      No matches.
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main */}
          <div className="lg:col-span-2">
            {selected ? (
              <PatientDetail
                patient={selected}
                onUpdate={updatePatient}
                onDelete={deletePatient}
              />
            ) : (
              <Card className="h-full">
                <CardHeader>
                  <CardTitle>Welcome</CardTitle>
                  <CardDescription>
                    Select a patient from the left, or add a new one.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <Tabs defaultValue="howto" className="w-full">
                    <TabsList>
                      <TabsTrigger value="howto">How to use</TabsTrigger>
                      <TabsTrigger value="fields">Patient fields</TabsTrigger>
                      <TabsTrigger value="tips">Tips</TabsTrigger>
                    </TabsList>
                    <TabsContent value="howto" className="space-y-2">
                      <ol className="list-decimal ml-5 text-sm space-y-1">
                        <li>Use the search bar to find by name, MRN, phone, condition, or medication.</li>
                        <li>Click <span className="font-medium">New</span> to add a patient. Required: first/last name, DOB.</li>
                        <li>Open a patient to view summary, vitals trend, and encounters.</li>
                        <li>Use <span className="font-medium">Add encounter</span> to log vitals, diagnosis, and notes.</li>
                        <li>Toggle <span className="font-medium">Persist locally</span> to save to your browser.</li>
                        <li>Export/import JSON for demos.</li>
                      </ol>
                    </TabsContent>
                    <TabsContent value="fields" className="text-sm">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div>
                          <div className="font-medium">Demographics</div>
                          <div className="text-muted-foreground">Name, DOB, sex, contact, address</div>
                        </div>
                        <div>
                          <div className="font-medium">Clinical</div>
                          <div className="text-muted-foreground">Allergies, conditions, medications, notes, BMI</div>
                        </div>
                        <div>
                          <div className="font-medium">Encounters</div>
                          <div className="text-muted-foreground">Date/time, vitals (BP/HR/Temp/SpO₂), diagnosis, prescription, free-text notes</div>
                        </div>
                        <div>
                          <div className="font-medium">Utilities</div>
                          <div className="text-muted-foreground">Export/Import JSON, Print view, Dark mode</div>
                        </div>
                      </div>
                    </TabsContent>
                    <TabsContent value="tips" className="text-sm">
                      <ul className="list-disc ml-5 space-y-1 text-muted-foreground">
                        <li>Click a chip (allergy, condition, med) in the editor to remove it.</li>
                        <li>Use the search to quickly jump to a patient or filter by a medication.</li>
                        <li>Trend lines update automatically as you add encounters.</li>
                        <li>Everything is client-side. No data leaves your browser.</li>
                      </ul>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
