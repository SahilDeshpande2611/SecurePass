import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('token');
    const role = localStorage.getItem('role');

    console.log('Home.jsx: token:', token);
    console.log('Home.jsx: role:', role);

    if (!token) {
      console.log('Home.jsx: No token, redirecting to /login');
      navigate('/login');
      return;
    }

    // Redirect based on role
    switch (role) {
      case 'admin':
        console.log('Home.jsx: Role is admin, redirecting to /admin');
        navigate('/admin');
        break;
      case 'security':
        console.log('Home.jsx: Role is security, redirecting to /security');
        navigate('/security');
        break;
      case 'manager':
        console.log('Home.jsx: Role is manager, redirecting to /manager');
        break;
      default:
        console.log('Home.jsx: Invalid role, redirecting to /login');
        navigate('/login');
    }
  }, [navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-500 to-indigo-600">
      <div className="bg-white p-8 rounded-lg shadow-lg">
        <h2 className="text-2xl font-semibold text-gray-800">Redirecting...</h2>
        <p className="text-gray-600 mt-2">Please wait while we take you to your dashboard.</p>
      </div>
    </div>
  );
};

export default Home;