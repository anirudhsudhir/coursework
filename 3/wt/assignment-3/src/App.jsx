import { useState } from "react";
import Card from "./Card.jsx";

function App() {
  const patientsList = [
    {
      id: 1,
      name: "Luke",
      reason: "Routine checkup",
      date: "01-11-2024",
    },
    {
      id: 2,
      name: "Han",
      reason: "Fever",
      date: "05-11-2024",
    },
    {
      id: 3,
      name: "Vader",
      reason: "Surgery",
      date: "10-12-2024",
    },
  ];

  let buttonStates = [];
  for (let i = 0; i < patientsList.length; i++) {
    const [completed, setCompleted] = useState(false);
    buttonStates.push([completed, setCompleted]);
  }

  const [completedAppointments, setCompletedAppointments] = useState(0);

  let i = 0;
  const patients = patientsList.map((patient) => (
    <Card
      key={patient.id}
      patient={patient}
      completed={buttonStates[i][0]}
      setCompleted={buttonStates[i++][1]}
      completedAppointments={completedAppointments}
      setCompletedAppointments={setCompletedAppointments}
    ></Card>
  ));
  console.log(patients);

  return (
    <div
      style={{
        fontFamily: "system-ui",
      }}
    >
      <h1
        style={{
          textAlign: "center",
        }}
      >
        Patient Appointments
      </h1>
      <div
        className="card-container"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          padding: "10px",
        }}
      >
        {patients}
      </div>
      <div
        className="counter"
        style={{
          fontWeight: "bold",
          fontSize: "1.5rem",
        }}
      >
        Total Completed Appointments: {completedAppointments}
      </div>
    </div>
  );
}

export default App;
