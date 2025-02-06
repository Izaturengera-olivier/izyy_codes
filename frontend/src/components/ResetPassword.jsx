import { useState } from "react";
import axios from "axios";

const ResetPassword = () => {
  const [formData, setFormData] = useState({ email: "", code: "", new_password: "" });
  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleReset = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:8000/verify-reset-code/", formData);
      setMessage(response.data.message);
    } catch (error) {
      setMessage(error.response?.data?.error || "Invalid code or password reset failed.");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
      <form onSubmit={handleReset} className="bg-white p-6 shadow-lg rounded">
        <h2 className="text-2xl font-bold">Reset Password</h2>
        <input
          type="email"
          name="email"
          placeholder="Enter your email"
          value={formData.email}
          onChange={handleChange}
          className="w-full p-2 border rounded mt-2"
          required
        />
        <input
          type="text"
          name="code"
          placeholder="Enter verification code"
          value={formData.code}
          onChange={handleChange}
          className="w-full p-2 border rounded mt-2"
          required
        />
        <input
          type="password"
          name="new_password"
          placeholder="Enter new password"
          value={formData.new_password}
          onChange={handleChange}
          className="w-full p-2 border rounded mt-2"
          required
        />
        <button type="submit" className="w-full bg-blue-500 text-white py-2 rounded mt-3">
          Reset Password
        </button>
        {message && <p className="text-green-500 mt-3">{message}</p>}
      </form>
    </div>
  );
};

export default ResetPassword;
