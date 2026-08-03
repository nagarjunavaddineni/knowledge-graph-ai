"""Unit tests for the reusable Neo4j client."""

from unittest.mock import MagicMock, patch

from app.database.neo4j_client import Neo4jClient


@patch("app.database.neo4j_client.GraphDatabase.driver")
def test_client_creates_driver(mock_driver_factory: MagicMock) -> None:
    """The client should create a driver with supplied credentials."""
    mock_driver = MagicMock()
    mock_driver_factory.return_value = mock_driver

    client = Neo4jClient(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
        database="neo4j",
    )

    mock_driver_factory.assert_called_once_with(
        "bolt://localhost:7687",
        auth=("neo4j", "test-password"),
    )

    client.close()


@patch("app.database.neo4j_client.GraphDatabase.driver")
def test_verify_connection_success(
    mock_driver_factory: MagicMock,
) -> None:
    """Connection verification should return True on success."""
    mock_driver = MagicMock()
    mock_driver_factory.return_value = mock_driver

    client = Neo4jClient(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
    )

    result = client.verify_connection()

    assert result is True
    mock_driver.verify_connectivity.assert_called_once_with()

    client.close()


@patch("app.database.neo4j_client.GraphDatabase.driver")
def test_execute_query_returns_dictionaries(
    mock_driver_factory: MagicMock,
) -> None:
    """Query results should be converted into dictionaries."""
    mock_driver = MagicMock()
    mock_record = MagicMock()
    mock_summary = MagicMock()

    mock_record.data.return_value = {"connection_test": 1}
    mock_summary.result_available_after = 2

    mock_driver.execute_query.return_value = (
        [mock_record],
        mock_summary,
        ["connection_test"],
    )
    mock_driver_factory.return_value = mock_driver

    client = Neo4jClient(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
    )

    result = client.execute_query(
        "RETURN $value AS connection_test",
        {"value": 1},
    )

    assert result == [{"connection_test": 1}]

    mock_driver.execute_query.assert_called_once_with(
        "RETURN $value AS connection_test",
        parameters_={"value": 1},
        database_="neo4j",
    )

    client.close()


@patch("app.database.neo4j_client.GraphDatabase.driver")
def test_context_manager_closes_driver(
    mock_driver_factory: MagicMock,
) -> None:
    """Leaving the context manager should close the driver."""
    mock_driver = MagicMock()
    mock_driver_factory.return_value = mock_driver

    with Neo4jClient(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
    ):
        pass

    mock_driver.close.assert_called_once_with()