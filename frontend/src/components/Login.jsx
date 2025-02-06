
import { useState } from "react";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";

const Login = () => {
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [message, setMessage] = useState("");
  const [token, setToken] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://127.0.0.1:8000/api/login/", formData);
      localStorage.setItem("access_token", res.data.tokens.access);
      setToken(res.data.tokens.access);
      setMessage("Login successful!");
      navigate("/"); // Redirect to home page (using relative path)
    } catch (error) {
      setMessage(error.response?.data?.error || "Invalid credentials");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white p-6 rounded shadow-lg w-96">
        <h2 className="text-2xl font-bold mb-4">Login</h2>
        {message && <p className={token ? "text-green-500" : "text-red-500"}>{message}</p>}
        <form onSubmit={handleSubmit} className="space-y-3">
          <input
            type="text"
            name="username"
            placeholder="Username"
            className="w-full p-2 border rounded"
            onChange={handleChange}
            required
          />
          <input
            type="password"
            name="password"
            placeholder="Password"
            className="w-full p-2 border rounded"
            onChange={handleChange}
            required
          />
          <button type="submit" className="w-full bg-blue-500 text-white p-2 rounded">
            Login
          </button>
        </form>

        <div className="mt-4 text-center"> {/* Added container for signup/reset links */}
          <p>
            Don't have an account? <Link to="/signup" className="text-blue-500 hover:underline">Sign Up</Link>
          </p>
          <p>
            Forgot your password? <Link to="/forgot-password" className="text-blue-500 hover:underline">Reset Password</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;