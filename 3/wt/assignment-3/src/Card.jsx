function Card({
  patient,
  completed,
  setCompleted,
  completedAppointments,
  setCompletedAppointments,
}) {
  const handleCompletedClick = () => {
    setCompleted(true);
    setCompletedAppointments(completedAppointments + 1);
  };

  return (
    <div
      className="card"
      style={{
        border: "2px solid black",
        borderRadius: "3px",
        padding: "10px",
        display: "flex",
        flexDirection: "column",
        gap: "10px",
      }}
    >
      <div
        className="card-header"
        style={{
          fontWeight: "bold",
        }}
      >
        {patient.name}
      </div>
      <div
        className="card-reason"
        style={
          completed
            ? {
                textDecoration: "line-through",
              }
            : {
                textDecoration: "none",
              }
        }
      >
        {patient.reason}
      </div>
      <div className="card-date">{patient.date}</div>
      <div className="card-status">
        Status: {!completed ? "scheduled" : "completed"}
      </div>
      {!completed && (
        <button
          className="card-reason"
          onClick={handleCompletedClick}
          style={{
            backgroundColor: "lightblue",
            width: "10vw",
            border: "2px solid blue",
            borderRadius: "3px",
            fontSize: "0.9rem",
          }}
        >
          Mark as completed
        </button>
      )}
    </div>
  );
}

export default Card;
