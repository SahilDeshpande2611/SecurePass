import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import Login from './components/Login';
import Register from './components/Register';
import AdminDashboard from './pages/AdminDashboard';
import SecurityDashboard from './pages/SecurityDashboard';
import ManagerDashboard from './pages/ManagerDashboard';

// Utility function for role-based redirection (same as in Login.jsx)
const redirectToDashboard = (role, navigate) => {
  console.log(`Redirecting user with role: ${role}`);
  switch (role) {
    case 'admin':
      navigate('/admin');
      break;
    case 'security':
      navigate('/security');
      break;
    case 'manager':
      navigate('/manager');
      break;
    default:
      console.error(`Invalid role: ${role}`);
      localStorage.clear();
      navigate('/login');
      break;
  }
};

// Protected Route Component
const ProtectedRoute = ({ element, allowedRole }) => {
  const navigate = useNavigate();
  const token = localStorage.getItem('token');
  const role = localStorage.getItem('role');

  useEffect(() => {
    console.log(`ProtectedRoute check - token: ${!!token}, role: ${role}, allowedRole: ${allowedRole}`);
    if (!token || !role) {
      console.warn('No token or role found, redirecting to login');
      navigate('/login');
      return;
    }

    if (role !== allowedRole) {
      console.warn(`Role ${role} not allowed for this route, redirecting based on role`);
      redirectToDashboard(role, navigate);
    }
  }, [token, role, allowedRole, navigate]);

  if (!token || !role || role !== allowedRole) {
    return null; // Prevent rendering until redirection happens
  }

  return element;
};

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/admin"
          element={<ProtectedRoute element={<AdminDashboard />} allowedRole="admin" />}
        />
        <Route
          path="/security"
          element={<ProtectedRoute element={<SecurityDashboard />} allowedRole="security" />}
        />
        <Route
          path="/manager"
          element={<ProtectedRoute element={<ManagerDashboard />} allowedRole="manager" />}
        />
        <Route path="/" element={<Navigate to="/login" />} />
      </Routes>
    </Router>
  );
}

export default App;